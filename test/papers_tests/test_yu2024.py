import numpy as np
import torch
from torch import nn

from abus_classification.papers.yu2024 import YuResNet, train_run
from abus_classification.papers.yu2024.pipeline import evaluate, tumor_slices, votes


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
