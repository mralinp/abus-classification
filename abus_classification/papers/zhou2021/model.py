"""Multi-task V-Net with the multi-scale classification branch (CMSVNet) of Zhou et al., Med. Image Anal. 2021."""
import torch
from torch import nn
from torch.nn import functional as F


def block(i, o, n=2):
    layers = []
    for k in range(n):
        layers += [nn.Conv3d(i if k == 0 else o, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU()]
    return nn.Sequential(*layers)


class MultiTaskVNet(nn.Module):
    def __init__(self, widths=(16, 32, 64, 128), scales=(4, 5, 6)):
        super().__init__()
        self.scales = scales                                                       # stages fed to the classifier
        chans = (1, *widths)
        self.enc = nn.ModuleList(block(chans[k], chans[k + 1]) for k in range(4))
        self.down = nn.ModuleList(nn.Conv3d(c, c, 2, 2) for c in widths)           # strided conv, not pooling
        self.mid = block(widths[-1], 2 * widths[-1], 3)                            # Stage 5
        self.up = nn.ModuleList(nn.ConvTranspose3d(c * 2, c, 2, 2) for c in reversed(widths))
        self.dec = nn.ModuleList(block(2 * c, c) for c in reversed(widths))
        self.seg = nn.Conv3d(widths[0], 1, 1)
        width = {4: widths[-1], 5: 2 * widths[-1], 6: widths[-1]}
        self.cls = nn.Sequential(nn.Linear(sum(width[k] for k in scales), 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))

    def forward(self, x):
        skips = []
        for e, d in zip(self.enc, self.down):
            x = e(x)
            skips.append(x)
            x = d(x)
        mid = self.mid(x)
        x, dec0 = mid, None
        for k, (u, d) in enumerate(zip(self.up, self.dec)):
            x = d(torch.cat([u(x), skips[-1 - k]], 1))
            dec0 = x if k == 0 else dec0
        stage = {4: skips[-1], 5: mid, 6: dec0}
        feats = [F.normalize(stage[k].mean((2, 3, 4)), dim=1) for k in self.scales]        # GAP + L2 norm per scale
        return self.seg(x).squeeze(1), self.cls(torch.cat(feats, 1)).squeeze(1)
