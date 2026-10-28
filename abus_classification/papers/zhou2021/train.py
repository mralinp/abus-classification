"""Zhou et al., Med. Image Anal. 70:101918 (2021): iterative feature-refining multi-task V-Net (CMSVNetIter).

    uv run python reproduce_zhou2021.py --epochs 100     # needs data/eswa2025.npz (see reproduce_eswa2025.py)

Paper details followed (Sec. 3-4): V-Net with strided 2x2x2 down-sampling convs, BN + ReLU; classification branch
on Stage 4 (deepest encoder), Stage 5 (bottleneck) and Stage 6 (first decoder) -> GAP -> L2 norm (Eq. 1) -> concat
-> 2 FC -> softmax; iterative refinement with N=2 (input of pass 2 = volume + pass-1 probability map, Alg. 1, one
Adam step per pass); class-weighted focal loss (gamma=2, Eq. 2-3) + Dice loss (Eq. 4), L = 0.3 L_cls + 0.7 L_seg;
Adam 1e-4 with decay, batch size 2.
Deviations: patches are our cached 128x64x80 voxels at 0.476 mm (paper 128x64x128 at 0.5 mm); augmentation is
random flips + isotropic scaling (no rotation / elastic); epochs, decay schedule and FC width are not given
(100 epochs, cosine, 256). Protocol: train on Train+Validation (130), report on the official Test split (70).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from abus_classification.data.lesion_cache import load_raw
from abus_classification.papers.zhou2021.losses import dice_loss, focal_loss
from abus_classification.papers.zhou2021.model import MultiTaskVNet
from abus_classification.protocol import DEVICE, auc_ci, metrics


def augment(v, m):
    """Random flips on all three axes and isotropic scaling in [0.8, 1.2], applied identically to volume and mask."""
    for ax in (1, 2, 3):
        if np.random.rand() < 0.5:
            v, m = v.flip(ax), m.flip(ax)
    s = torch.empty(len(v), device=v.device).uniform_(0.8, 1.2)
    theta = torch.cat([torch.diag_embed(s[:, None].expand(-1, 3)), torch.zeros(len(v), 3, 1, device=v.device)], 2)
    grid = F.affine_grid(theta, (len(v), 1, *v.shape[1:]), align_corners=False)
    warp = lambda t, mode: F.grid_sample(t[:, None], grid, mode=mode, align_corners=False)[:, 0]
    return warp(v, "bilinear"), (warp(m, "bilinear") > 0.5).float()


def train_eval(X, y, tv, te, epochs, seed, lam=0.3, iters=2, ckpt_dir="."):
    torch.manual_seed(seed), np.random.seed(seed)
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32))
    vol, mask, lab = t(X["vol"][tv]), t(X["mask"][tv]), t(y[tv])
    w_m, w_n = (y[tv] == 0).mean(), (y[tv] == 1).mean()             # Eq. 3: w_m = N_n / (N_m + N_n)
    net = MultiTaskVNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), 1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ckpt, start = Path(ckpt_dir) / f"zhou2021_seed{seed}.pt", 0
    if ckpt.exists():                                                # resume after a crash / re-evaluate a finished run
        state = torch.load(ckpt, map_location=DEVICE)
        net.load_state_dict(state["net"]), opt.load_state_dict(state["opt"]), sched.load_state_dict(state["sched"])
        start = state["epoch"]
        print(f"seed {seed}: resumed from {ckpt} at epoch {start}", flush=True)
    for ep in range(start, epochs):
        net.train()
        for idx in torch.randperm(len(lab)).split(2):
            if len(idx) < 2:                                         # BatchNorm needs more than one sample
                continue
            v, m = augment(vol[idx].to(DEVICE), mask[idx].to(DEVICE))
            l, prob = lab[idx].to(DEVICE), torch.zeros_like(v)
            for _ in range(iters):                                   # Alg. 1: one Adam step per iteration
                seg, logit = net(v[:, None] + prob[:, None])
                loss = lam * focal_loss(logit, l, w_m, w_n) + (1 - lam) * dice_loss(seg, m)
                opt.zero_grad(), loss.backward(), opt.step()
                prob = torch.sigmoid(seg).detach()
        sched.step()
        if ep % 10 == 9 or ep + 1 == epochs:
            print(f"seed {seed} epoch {ep + 1} loss {loss.item():.3f}", flush=True)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1), ckpt)
    net.eval()
    probs, dices = [], []
    with torch.no_grad():
        for idx in np.array_split(np.arange(len(te)), max(1, len(te) // 2)):        # small batches: 70 at once OOMs
            v, m = t(X["vol"][te[idx]]).to(DEVICE), t(X["mask"][te[idx]]).to(DEVICE) > 0.5
            prob = torch.zeros_like(v)
            for _ in range(iters):
                seg, logit = net(v[:, None] + prob[:, None])
                prob = torch.sigmoid(seg)
            pred = prob > 0.5
            dices.append((2 * (pred & m).sum((1, 2, 3)).float() / (pred.sum((1, 2, 3)) + m.sum((1, 2, 3)))).cpu())
            probs.append(torch.sigmoid(logit).cpu())
    return torch.cat(probs).numpy(), torch.cat(dices).mean().item()


def main(epochs, seeds=(0, 1, 2), ckpt_dir="."):
    X, tv, te = load_raw()
    y = X["y"]
    runs = []
    for s in seeds:
        runs.append(train_eval(X, y, tv, te, epochs, s, ckpt_dir=ckpt_dir))
        np.save(Path(ckpt_dir) / f"zhou2021_probs_seed{s}.npy", runs[-1][0])                  # keep each seed even if a later one fails
    rows = {f"seed {s}": {**metrics(y[te], p), "Dice": d} for s, (p, d) in zip(seeds, runs)}
    mean = np.mean([p for p, _ in runs], 0)
    rows["mean of seeds"] = {**metrics(y[te], mean), "Dice": np.mean([d for _, d in runs])}
    table = pd.DataFrame(rows).T.round(4)
    print(table.to_string())
    print("AUC 95%% CI of the seed ensemble: %.3f-%.3f" % tuple(auc_ci(y[te], mean)))
    return table
