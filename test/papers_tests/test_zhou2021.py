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
