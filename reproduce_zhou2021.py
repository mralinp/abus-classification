"""Zhou et al., Med. Image Anal. 2021: multi-task V-Net for ABUS tumour segmentation + classification.

    uv run python reproduce_zhou2021.py --epochs 60     # needs data/eswa2025.npz (see reproduce_eswa2025.py)

Segmentation encoder-decoder plus a light multi-scale classifier on its encoder / bottleneck / decoder
features. Iterative refinement: the first pass sees (volume, empty map); the second sees (volume, first-pass
probability map). Both passes get a segmentation loss, the second also the classification loss.
Train on Train+Validation (130 volumes), report on the official Test split (70), malignant = positive.
The paper's loss weights, patch size and schedule are not in the abstract; those below are our choices.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from reproduce_eswa2025 import DEVICE, load, metrics


def block(i, o, n=2):
    layers = []
    for k in range(n):
        layers += [nn.Conv3d(i if k == 0 else o, o, 3, padding=1), nn.InstanceNorm3d(o), nn.ReLU()]
    return nn.Sequential(*layers)


class MultiTaskVNet(nn.Module):
    def __init__(self, widths=(16, 32, 64, 128)):
        super().__init__()
        chans = (2, *widths)
        self.enc = nn.ModuleList(block(chans[k], chans[k + 1]) for k in range(4))
        self.mid = block(widths[-1], 2 * widths[-1], 3)
        self.up = nn.ModuleList(nn.ConvTranspose3d(c * 2, c, 2, 2) for c in reversed(widths))
        self.dec = nn.ModuleList(block(2 * c, c) for c in reversed(widths))
        self.seg = nn.Conv3d(widths[0], 1, 1)
        feat = widths[-1] + 2 * widths[-1] + widths[-1]            # deepest encoder + bottleneck + first decoder
        self.cls = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = F.max_pool3d(x, 2)
        mid = self.mid(x)
        x, dec0 = mid, None
        for k, (u, d) in enumerate(zip(self.up, self.dec)):
            x = d(torch.cat([u(x), skips[-1 - k]], 1))
            dec0 = x if k == 0 else dec0
        gap = lambda t: t.mean((2, 3, 4))
        logit = self.cls(torch.cat([gap(skips[-1]), gap(mid), gap(dec0)], 1)).squeeze(1)
        return self.seg(x).squeeze(1), logit

    def iterate(self, vol, passes=2):
        """Run the iterative scheme; returns [(seg_logit, cls_logit)] per pass."""
        prob, out = torch.zeros_like(vol), []
        for _ in range(passes):
            seg, logit = self(torch.stack([vol, prob], 1))
            out.append((seg, logit))
            prob = torch.sigmoid(seg).detach()
        return out


def dice_loss(seg, mask):
    p = torch.sigmoid(seg)
    return 1 - (2 * (p * mask).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1)


def train_eval(X, y, tv, te, epochs, seed, alpha=0.5):
    torch.manual_seed(seed), np.random.seed(seed)
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32))
    vol, mask, lab = t(X["vol"][tv]), t(X["mask"][tv]), t(y[tv])
    net = MultiTaskVNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(len(lab)).split(4):
            v, m, l = vol[idx].to(DEVICE), mask[idx].to(DEVICE), lab[idx].to(DEVICE)
            for ax in (1, 3):                                        # random flips
                if np.random.rand() < 0.5:
                    v, m = v.flip(ax), m.flip(ax)
            passes = net.iterate(v)
            seg = sum(F.binary_cross_entropy_with_logits(s, m) + dice_loss(s, m).mean() for s, _ in passes)
            loss = alpha * seg + (1 - alpha) * F.binary_cross_entropy_with_logits(passes[-1][1], l)
            opt.zero_grad(), loss.backward(), opt.step()
        sched.step()
        if ep % 10 == 9:
            print(f"seed {seed} epoch {ep + 1} loss {loss.item():.3f}", flush=True)
    net.eval()
    with torch.no_grad():
        v, m = t(X["vol"][te]).to(DEVICE), t(X["mask"][te])
        seg, logit = net.iterate(v)[-1]
        dice = 1 - dice_loss(seg.cpu(), m).mean().item()
        return torch.sigmoid(logit).cpu().numpy(), dice


def auc_ci(y, p, n=1000, seed=0):
    rng, aucs = np.random.default_rng(seed), []
    while len(aucs) < n:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])


def main(epochs, seeds=(0, 1, 2)):
    X, y, tv, te = load()
    runs = [train_eval(X, y, tv, te, epochs, s) for s in seeds]
    rows = {f"seed {s}": {**metrics(y[te], p), "Dice": d} for s, (p, d) in zip(seeds, runs)}
    mean = np.mean([p for p, _ in runs], 0)
    rows["mean of seeds"] = {**metrics(y[te], mean), "Dice": np.mean([d for _, d in runs])}
    table = pd.DataFrame(rows).T.round(4)
    print(table.to_string())
    print("AUC 95%% CI of the seed ensemble: %.3f-%.3f" % tuple(auc_ci(y[te], mean)))
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    main(a.epochs, a.seeds).to_csv("zhou2021_results.csv")
