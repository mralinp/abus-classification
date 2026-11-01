import numpy as np
import torch

from abus_classification import challenge
from abus_classification.papers.luo2025_t1 import CLSNet, DETNet, SEGNet
from abus_classification.papers.luo2025_t1 import pipeline as p


def test_model_outputs():
    x = torch.zeros(2, 1, 32, 16, 32)
    assert SEGNet().eval()(x).shape == (2, 32, 16, 32)
    assert DETNet().eval()(x).shape == (2, 7, 16, 8, 16)
    assert CLSNet().eval()(x).shape == (2, 2)


def test_box_is_in_continuous_voxels():
    m = np.zeros((8, 8, 8), bool)
    m[2:4, 3, 1:6] = True
    assert p.box_of(m).tolist() == [3, 3.5, 3.5, 2, 1, 5]


def test_crop_pads_outside_the_volume():
    v = torch.arange(1, 28).reshape(3, 3, 3)
    out = p.crop(v, (0, 1.5, 1.5), (2, 3, 3))
    assert (out[0] == 0).all() and torch.equal(out[1], v[0])


def test_heat_focal_by_hand():
    logit, heat = torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.5])
    # positive: -(1 - .5)^2 ln .5; other: -(1 - .5)^4 .5^2 ln .5, over one positive
    assert np.isclose(p.heat_focal(logit, heat).item(), (0.25 + 1 / 64) * np.log(2), atol=1e-5)


def test_decode_inverts_the_targets():
    boxes = torch.tensor([[20.0, 10.0, 30.0, 12.0, 6.0, 18.0]])
    heat, cells = p.heat_targets(boxes, (16, 8, 24))
    assert (heat == 1).sum() == 1 and heat.max() == 1
    offset = (boxes[:, :3].view(-1, 3, 1, 1, 1) - cells[None]) / 8
    size = boxes[:, 3:].log().view(-1, 3, 1, 1, 1).expand_as(offset)
    logit = (heat * 20 - 10)[:, None]
    found, scores = p.decode(torch.cat([logit, offset, size], 1), k=1)
    assert torch.allclose(found[0, 0], boxes[0], atol=1e-4) and scores[0, 0] > 0.99


def test_det_batch_is_on_the_training_device(monkeypatch):
    monkeypatch.setattr(p, "DEVICE", "meta")
    x, boxes = p.det_batch([torch.zeros(20, 10, 20, dtype=torch.uint8)], np.zeros((1, 6), np.float32))
    assert x.device.type == boxes.device.type == "meta"


def test_lesion_dice_by_hand():
    class Perfect(torch.nn.Module):
        def forward(self, x):
            return (x[:, 0] - 0.5) * 20
    v = torch.zeros(20, 10, 20, dtype=torch.uint8)
    v[8:12, 3:7, 8:12] = 255
    data = dict(vol=[v], mask=[v > 0], box=np.array([[10, 5, 10, 4, 4, 4]], np.float32))
    assert p.lesion_dice(Perfect(), data, [0], (16, 8, 16)) == 1


def test_a_crash_reaches_the_progress_log(tmp_path):
    import pytest
    with pytest.raises(Exception):
        p.run_job(dict(task="det", seed=0, out_dir=str(tmp_path), cache=str(tmp_path / "missing.npz"), max_seconds=1))
    assert "det_seed0 failed" in (tmp_path / "luo2025_t1_progress.log").read_text()


def test_windows_cover_the_volume():
    for shape in ((300, 80, 257), (128, 64, 128), (100, 40, 90)):
        cover = np.zeros(shape, bool)
        for corner in p.windows(shape, p.PATCH):
            cover[tuple(slice(c, c + s) for c, s in zip(corner, p.PATCH))] = True
        assert cover.all()


def test_challenge_box_and_upsampling_match_the_original_grid():
    m = np.zeros((40, 80, 20), bool)
    m[8:24, 16:48, 4:12] = True
    small = m[::4, ::8, ::2]
    assert np.array_equal(p.to_original(small, m.shape), m)
    box = p.challenge_box(p.box_of(small), small.shape, m.shape)
    assert challenge.iou_3d(box, challenge.mask_box(m)) > 0.75      # mask_box is one voxel short per axis


def cases():
    """Twelve volumes with a bright cube, malignant when it is bigger. Eight fit, two validate, two test."""
    rng, out = np.random.default_rng(0), []
    for i in range(12):
        y = i % 2
        v, m = rng.integers(0, 40, (48, 24, 48), dtype=np.uint8), np.zeros((48, 24, 48), bool)
        half = 8 if y else 4
        c = rng.integers(10, 14), rng.integers(9, 15), rng.integers(10, 38)
        c = (c[0] + 24 * (i % 3 == 0), c[1], c[2])
        m[c[0] - half:c[0] + half, c[1] - 4:c[1] + 4, c[2] - half:c[2] + half] = True
        v[m] = 220
        out.append(dict(vol=v, mask=m, y=y, split=0 if i < 8 else 1 if i < 10 else 2, source=i, shape=(96, 96, 48), box=p.box_of(m)))
    return out


def test_all_tasks_train_and_evaluate(tmp_path):
    made = cases()
    p.write_cache(made, tmp_path / "c.npz")
    for task, epochs in (("det", 60), ("seg", 15), ("cls", 10)):
        p.run_job(dict(task=task, seed=0, out_dir=str(tmp_path), cache=str(tmp_path / "c.npz"), max_seconds=600, patch=(32, 16, 32),
                       overrides=dict(steps=10, epochs=epochs, patience=epochs, lr=1e-3)))
    data = p.load_cache(tmp_path / "c.npz")
    assert p.dice(p.segment(_loaded(tmp_path), data["vol"][10], (32, 16, 32)), made[10]["mask"]) > 0.5

    def big(i):                                                       # the original grid: twice the cache on axes 0 and 1
        return np.repeat(np.repeat(made[i]["mask"], 2, 0), 2, 1)
    table = p.evaluate(data, (0,), tmp_path, big)
    row = table.loc["seed 0"]
    assert row.DICE > 0.5 and row.FROC > 0.5 and row.empty_cases == 0
    assert (tmp_path / "luo2025_t1_benchmark.csv").exists() and np.load(tmp_path / "luo2025_t1_seg_cases_seed0.npy").shape == (2, 2)


def test_a_stopped_training_resumes_from_its_checkpoint(tmp_path):
    made = cases()
    p.write_cache(made, tmp_path / "c.npz")
    job = dict(task="det", seed=0, out_dir=str(tmp_path), cache=str(tmp_path / "c.npz"), max_seconds=600, overrides=dict(steps=1, epochs=2, patience=9))
    p.run_job(dict(job, overrides=dict(steps=1, epochs=1, patience=9)))
    (tmp_path / "luo2025_t1_det_seed0.npz").unlink()
    ckpt = tmp_path / "luo2025_t1_det_seed0.ckpt"
    net = DETNet()
    opt = torch.optim.Adam(net.parameters())
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max")
    state = dict(epoch=1, best={k: v.clone() for k, v in net.state_dict().items()}, best_score=0.0, best_epoch=1, seconds=0.0, history=[dict(epoch=1)], done=False)
    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), plateau=plateau.state_dict(), numpy=np.random.get_state(), state=state), ckpt)
    p.run_job(job)
    assert "resumed after epoch 1" in (tmp_path / "luo2025_t1_progress.log").read_text()
    assert not ckpt.exists() and (tmp_path / "luo2025_t1_det_seed0.pt").exists()
    assert len((tmp_path / "luo2025_t1_det_seed0_history.csv").read_text().splitlines()) == 3


def test_a_past_deadline_leaves_no_predictions(tmp_path):
    p.write_cache(cases(), tmp_path / "c.npz")
    p.run_job(dict(task="det", seed=0, out_dir=str(tmp_path), cache=str(tmp_path / "c.npz"), max_seconds=600, deadline=1.0))
    assert not (tmp_path / "luo2025_t1_det_seed0.npz").exists()


def _loaded(path):
    net = SEGNet()
    net.load_state_dict(torch.load(path / "luo2025_t1_seg_seed0.pt"))
    return net.to(p.DEVICE)
