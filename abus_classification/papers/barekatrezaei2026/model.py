"""Dual-path 3D CNN of Barekatrezaei et al. (ESWA 2026): VOI and mask paths, LB features joined after GAP."""
import numpy as np
import torch
from torch import nn

from abus_classification.protocol import DEVICE


class DualPath(nn.Module):
    """Two conv paths (VOI, mask) -> concat -> GAP -> + LB eigenvalues -> dense -> sigmoid."""

    def __init__(self, n_lb, widths=(16, 32, 64, 128)):
        super().__init__()

        def path():
            layers, c = [], 1
            for w in widths:
                for _ in range(2):
                    layers += [nn.Conv3d(c, w, 3, padding=1), nn.InstanceNorm3d(w), nn.ReLU()]
                    c = w
                layers.append(nn.MaxPool3d(2))
            return nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1), nn.Flatten())

        self.vol, self.mask = path(), path()
        self.head = nn.Sequential(nn.Linear(2 * widths[-1] + n_lb, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, vol, mask, lb):
        return self.head(torch.cat([self.vol(vol), self.mask(mask), lb], 1)).squeeze(1)


def fit_cnn(tr, te_sets, epochs, seed):
    """Train on indices `tr` of (vol, mask, lb, y); return sigmoid outputs for each set in `te_sets`."""
    torch.manual_seed(seed)
    net = DualPath(tr[2].shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32)).to(DEVICE)
    vol, mask, lb, y = map(t, tr)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(y)).split(8):
            v, m = vol[idx][:, None], mask[idx][:, None]
            for ax in (2, 4):                                # random flips (axis 0 and 2 of the patch)
                if np.random.rand() < 0.5:
                    v, m = v.flip(ax), m.flip(ax)
            loss = nn.functional.binary_cross_entropy_with_logits(net(v, m, lb[idx]), y[idx])
            opt.zero_grad(), loss.backward(), opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        return [torch.sigmoid(net(t(v)[:, None], t(m)[:, None], t(l))).cpu().numpy() for v, m, l in te_sets]
