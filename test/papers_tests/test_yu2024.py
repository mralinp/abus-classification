import numpy as np
import torch
from torch import nn

from abus_classification.papers.yu2024 import YuResNet, train_run
from abus_classification.papers.yu2024.pipeline import evaluate, plan, tumor_slices, votes
from abus_classification.protocol import run_on_gpus


def test_slices_keep_only_tumor_and_count_its_voxels():
    vol, mask = np.zeros((40, 30, 10), np.uint8), np.zeros((40, 30, 10), bool)
    vol[10:20, 5:15, 3] = 200
    mask[10:20, 5:15, 3] = True                                       # 100 voxels
    mask[12:14, 7:9, 6] = True                                        # 4 voxels
    vol[12:14, 7:9, 6] = 90
    whole, crop, voxels = tumor_slices(vol, mask, size=8)
    assert whole.shape == crop.shape == (2, 8, 8) and voxels.tolist() == [100, 4]
    assert (crop[0] == 200).all() and (crop[1] == 90).all() and whole[0].min() == 0


def test_votes_by_hand():
    p = np.array([0.9, 0.2, 0.2, 0.7, 0.6])
    voxels = np.array([800, 100, 100, 1, 1])
    case = np.array([0, 0, 0, 1, 1])
    cases, hard, soft = votes(p, voxels, case)
    assert cases.tolist() == [0, 1] and hard.tolist() == [1 / 3, 1.0]
    # one big malignant slice outvotes two small benign ones: (720 + 20 + 20) / 1000
    assert np.isclose(soft[0], 0.76) and np.isclose(soft[1], 0.65)


def test_model_output():
    net = YuResNet(pretrained=False).eval()
    assert net(torch.zeros(2, 1, 224, 224)).shape == (2, 2)
    assert sum(p.numel() for p in net.parameters()) == 21_797_672 - 513_000 + 1_026       # torchvision ResNet34, 1000-class head swapped for two


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc(x.flatten(1))


def data():
    """Six cases of two slices, class visible in brightness; the last two are held out."""
    y = np.array([0, 1, 0, 1, 0, 1])
    case = np.repeat(np.arange(6), 2)
    whole = np.stack([np.full((4, 4), 200 if y[c] else 20, np.uint8) for c in case])
    return dict(whole=whole, crop=whole, voxels=np.ones(12), case=case, y=y, split=np.array([0, 0, 0, 0, 2, 2]))


def test_overfit_evaluate_and_finished_runs_are_loaded(tmp_path):
    d = data()
    run = lambda epochs: train_run(d, "resnet34", 0, [0, 1, 2, 3], [4, 5], "t", tmp_path, epochs=epochs, batch=4, lr=0.05, make=Tiny)
    prob = run(60)
    assert prob.shape == (4,) and prob[:2].max() < 0.5 < prob[2:].min()
    assert np.array_equal(run(1), prob)
    rows, y, hard, soft = evaluate(d, prob, [4, 5])
    assert y.tolist() == [0, 1] and hard.tolist() == [0, 1] and rows["soft"]["AUC"] == 1 and rows["slices"]["Acc"] == 1


def test_early_stopping_keeps_the_best_epoch_on_validation(tmp_path):
    d = data()
    prob = train_run(d, "resnet34", 0, [0, 1], [4, 5], "es", tmp_path, epochs=50, batch=4, lr=0.05, make=Tiny, val=[2, 3], patience=3)
    saved = np.load(tmp_path / "yu2024_es.npz")
    assert prob.shape == (4,) and saved["val_auc"] == 1 and saved["best_epoch"] >= 1
    history = (tmp_path / "yu2024_es_history.csv").read_text().splitlines()
    assert len(history) - 1 == saved["best_epoch"] + 3 and "val_auc" in history[0]      # stopped after the patience ran out
    assert not (tmp_path / "yu2024_es.ckpt").exists() and (tmp_path / "yu2024_progress.log").exists()


def test_a_stopped_run_resumes_from_its_checkpoint(tmp_path):
    import time
    d = data()
    run = lambda deadline, epochs: train_run(d, "resnet34", 0, [0, 1, 2, 3], [4, 5], "r", tmp_path, epochs=epochs, batch=4, lr=0.05, make=Tiny, deadline=deadline)
    assert run(time.time() + 1e-9, 5) is None                         # past the deadline before any epoch
    assert run(None, 3) is not None
    assert len((tmp_path / "yu2024_r_history.csv").read_text().splitlines()) == 4
    ckpt = {"net": Tiny().state_dict()}                                 # as a stop at epoch 2 leaves it
    (tmp_path / "yu2024_r.npz").unlink()
    d2 = data()
    opt = torch.optim.Adam(Tiny().parameters(), lr=0.05)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max")
    state = dict(epoch=2, best=None, best_auc=-1.0, best_epoch=0, since=0, history=[dict(epoch=1), dict(epoch=2)])
    torch.save(dict(**ckpt, opt=opt.state_dict(), plateau=plateau.state_dict(), numpy=np.random.get_state(), state=state), tmp_path / "yu2024_r.ckpt")
    train_run(d2, "resnet34", 0, [0, 1, 2, 3], [4, 5], "r", tmp_path, epochs=4, batch=4, lr=0.05, make=Tiny)
    assert [int(l.split(",")[0]) for l in (tmp_path / "yu2024_r_history.csv").read_text().splitlines()[1:]] == [1, 2, 3, 4]
    assert "resumed after epoch 2" in (tmp_path / "yu2024_progress.log").read_text()


def test_plan_counts_and_keeps_validation_out_of_fitting():
    y = np.array([0, 1] * 15 + [0, 1] * 5 + [0, 1] * 5)
    d = dict(y=y, split=np.array([0] * 30 + [1] * 10 + [2] * 10))
    es, paper = plan(d, ["resnet34", "resnet34_scratch"], (0, 1, 2)), plan(d, ["resnet34", "resnet34_scratch"], (0, 1, 2), False)
    assert len(es) == 3 + 5 + 3 and all(not set(r["fit"]) & set(r["val"]) and not set(r["test"]) & set(r["val"]) for r in es)
    assert all(r["name"].startswith("es_") for r in es) and all(r["val"] is None for r in paper) and len(paper[0]["fit"]) == 40


def test_jobs_run_in_spawned_workers():
    assert run_on_gpus(abs, [-2, 3]) == [2, 3]
