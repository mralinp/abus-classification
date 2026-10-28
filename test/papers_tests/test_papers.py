import numpy as np
import torch

from abus_classification.papers.barekatrezaei2026.model import DualPath
from abus_classification.papers.zhou2021.losses import dice_loss, focal_loss
from abus_classification.papers.zhou2021.model import MultiTaskVNet
from abus_classification.protocol import auc_ci, metrics


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


def test_dual_path_forward():
    out = DualPath(n_lb=5).eval()(torch.randn(2, 1, 16, 16, 16), torch.randn(2, 1, 16, 16, 16), torch.randn(2, 5))
    assert out.shape == (2,)


def test_protocol_metrics():
    y, p = np.array([1, 1, 0, 0]), np.array([0.9, 0.6, 0.4, 0.1])
    m = metrics(y, p)
    assert m["AUC"] == 1.0 and m["Sens"] == 1.0 and m["Spec"] == 1.0
    lo, hi = auc_ci(y, p, n=50)
    assert lo <= hi <= 1.0
