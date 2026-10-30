import numpy as np
import torch

from abus_classification.papers.wang2020 import WangCNN
from abus_classification.papers.wang2020.pipeline import MODES, lesion_patches, rotate_flip, train_run


def test_patches_crop_the_box_in_both_views():
    vol = np.zeros((60, 50, 40), np.uint8)
    vol[10:21, 5:36, 8:30] = 200                                     # the lesion fills the box
    patches = lesion_patches(vol, ((8, 5, 10), (29, 35, 20)), size=16)
    assert patches.shape == (4, 16, 16) and (patches == 200).all()


def test_slices_are_taken_at_one_and_two_thirds():
    vol = np.zeros((10, 31, 31), np.uint8)
    vol[:, 10] = 50                                                  # coronal plane y = 10
    vol[:, :, 20] = 90                                               # transverse plane x = 20
    p = lesion_patches(vol, ((0, 0, 0), (30, 30, 9)), size=8)
    assert p[0].min() >= 50 and p[3].min() >= 90 and p[1].min() == 0 and p[2].min() == 0


def test_view_combinations():
    assert len(MODES["cnn_a"]) == 8 and (0, 2, 3) in MODES["cnn_a"]           # coronal, transverse, transverse
    assert len(MODES["cnn_b"]) == 4 and all(len(c) == 6 for c in MODES["cnn_b"])
    assert MODES["coronal"] == [(0, 0, 0), (1, 1, 1)]


def test_model_shapes_and_head():
    x = torch.zeros(2, 3, 299, 299)
    a = WangCNN(3, pretrained=False).eval()
    assert a(x).shape == (2, 2) and a.fc[0].in_features == 2048 and [m.out_features for m in a.fc if hasattr(m, "out_features")] == [256, 128, 2]
    b = WangCNN(6, pretrained=False).eval()
    assert len(b.backbones) == 2 and b.fc[0].in_features == 4096 and b(torch.zeros(2, 6, 299, 299)).shape == (2, 2)


def test_augmentation_moves_channels_together():
    x = torch.arange(16.).reshape(4, 4).repeat(3, 3, 1, 1)
    out = rotate_flip(x)
    assert out.shape == x.shape and all((s[0] == s[1]).all() and (s[0] == s[2]).all() for s in out)


def test_finished_run_is_not_trained_again(tmp_path, monkeypatch):
    monkeypatch.setitem(MODES, "tiny", [(0, 2, 2)])
    monkeypatch.setattr("abus_classification.papers.wang2020.pipeline.WangCNN", lambda *a, **k: Tiny())
    rng = np.random.default_rng(0)
    data = {"patches": rng.integers(0, 255, (12, 4, 8, 8), dtype=np.uint8), "y": np.array([0, 1] * 6), "split": np.array([0] * 6 + [1] * 2 + [2] * 4)}
    p = train_run(data, "tiny", 0, tmp_path, max_epochs=2, batch=4)
    before = (tmp_path / "wang2020_tiny_seed0.pt").read_bytes()
    assert p.shape == (4,) and np.allclose(train_run(data, "tiny", 0, tmp_path), p)
    assert (tmp_path / "wang2020_tiny_seed0.pt").read_bytes() == before


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x.mean((2, 3)))
