import numpy as np
import torch

from abus_classification.papers.zhou2021.losses import dice_loss, focal_loss
from abus_classification.papers.zhou2021.model import MultiTaskVNet


def test_zhou_shapes_and_branch_width():
    net = MultiTaskVNet().eval()
    seg, logit = net(torch.randn(2, 1, 32, 16, 32))
    assert seg.shape == (2, 32, 16, 32) and logit.shape == (2,)
    assert net.cls[0].in_features == 4 * 128          # Stage 4 (128) + Stage 5 (256) + Stage 6 (128), Fig. 2b


def test_zhou_losses_by_hand():
    perfect = torch.full((1, 4, 4, 4), 20.0)
    assert dice_loss(perfect, torch.ones(1, 4, 4, 4)) < 1e-6
    # Eq. 2 with p = 0.5, gamma = 2, y = 1: -w_m * 0.25 * log(0.5)
    expected = 0.5 * 0.25 * np.log(2)
    assert abs(focal_loss(torch.zeros(1), torch.ones(1), 0.5, 0.5).item() - expected) < 1e-6


def test_zhou_finished_checkpoint_is_reused_not_overwritten(tmp_path):
    from abus_classification.papers.zhou2021.train import train_eval
    rng = np.random.default_rng(0)
    X = {"vol": rng.random((8, 16, 16, 16)).astype(np.float32), "mask": np.zeros((8, 16, 16, 16), np.float32)}
    X["mask"][:, 4:12, 4:12, 4:12] = 1
    y = np.array([0, 1] * 4)
    tv, te = np.arange(6), np.arange(6, 8)
    p, _ = train_eval(X, y, tv, te, epochs=1, seed=0, ckpt_dir=tmp_path)
    ckpt = tmp_path / "zhou2021_seed0.pt"
    before = ckpt.read_bytes()
    p2, _ = train_eval(X, y, tv, te, epochs=1, seed=0, ckpt_dir=tmp_path)    # finished: evaluate only
    assert ckpt.read_bytes() == before and np.allclose(p, p2)
    train_eval(X, y, tv[:4], te, epochs=1, seed=0, ckpt_dir=tmp_path, va=tv[4:])
    assert ckpt.read_bytes() == before and (tmp_path / "zhou2021_seed0_val.pt").exists()
    assert {"focal", "dice_loss", "train_auc", "val_auc"} <= set(open(tmp_path / "zhou2021_seed0_val_history.csv").readline().strip().split(","))
    assert np.load(tmp_path / "zhou2021_seed0_test.npz")["seg_mid"].shape == (2, 16, 16)
