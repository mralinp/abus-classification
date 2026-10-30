import numpy as np
import torch

from abus_classification.papers.barekatrezaei2026.model import DualPath
from abus_classification.protocol import auc_ci, metrics


def test_dual_path_forward():
    out = DualPath(n_lb=5).eval()(torch.randn(2, 1, 16, 16, 16), torch.randn(2, 1, 16, 16, 16), torch.randn(2, 5))
    assert out.shape == (2,)


def test_protocol_metrics():
    y, p = np.array([1, 1, 0, 0]), np.array([0.9, 0.6, 0.4, 0.1])
    m = metrics(y, p)
    assert m["AUC"] == 1.0 and m["Sens"] == 1.0 and m["Spec"] == 1.0
    lo, hi = auc_ci(y, p, n=50)
    assert lo <= hi <= 1.0
