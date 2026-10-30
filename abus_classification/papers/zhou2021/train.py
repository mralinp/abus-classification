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
Diagnostics: per-epoch focal and Dice terms, train AUC and p(malignant) range go to `zhou2021_seed<s>_history.csv`;
`val=True` holds out 20% of Train+Validation (never Test) and adds val AUC / Dice per epoch.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
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


def predict(net, X, idx, iters=2):
    """p(malignant), per-case Dice and the coronal mid-plane (depth index 32) of the tumor probability map."""
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32))
    net.eval()
    probs, dices, mids = [], [], []
    with torch.no_grad():
        for b in np.array_split(np.arange(len(idx)), max(1, len(idx) // 2)):         # small batches: 70 at once OOMs
            v, m = t(X["vol"][idx[b]]).to(DEVICE), t(X["mask"][idx[b]]).to(DEVICE) > 0.5
            prob = torch.zeros_like(v)
            for _ in range(iters):
                seg, logit = net(v[:, None] + prob[:, None])
                prob = torch.sigmoid(seg)
            pred = prob > 0.5
            dices.append((2 * (pred & m).sum((1, 2, 3)).float() / (pred.sum((1, 2, 3)) + m.sum((1, 2, 3)))).cpu())
            probs.append(torch.sigmoid(logit).cpu())
            mids.append(prob[:, :, v.shape[2] // 2].half().cpu())
    return torch.cat(probs).numpy(), torch.cat(dices).numpy(), torch.cat(mids).numpy()


def train_eval(X, y, tv, te, epochs, seed, lam=0.3, iters=2, ckpt_dir=".", va=None, scales=(4, 5, 6), name=""):
    torch.manual_seed(seed), np.random.seed(seed)
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32))
    vol, mask, lab = t(X["vol"][tv]), t(X["mask"][tv]), t(y[tv])
    w_m, w_n = (y[tv] == 0).mean(), (y[tv] == 1).mean()             # Eq. 3: w_m = N_n / (N_m + N_n)
    net = MultiTaskVNet(scales=scales).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), 1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    # val and ablation runs have their own files: they never touch the protocol checkpoints
    tag = f"zhou2021_{name + '_' if name else ''}seed{seed}" + ("_val" if va is not None else "")
    ckpt, start, history = Path(ckpt_dir) / f"{tag}.pt", 0, []
    if ckpt.exists():                                                # resume after a crash / re-evaluate a finished run
        state = torch.load(ckpt, map_location=DEVICE)
        net.load_state_dict(state["net"]), opt.load_state_dict(state["opt"]), sched.load_state_dict(state["sched"])
        start, history = state["epoch"], state.get("history", [])     # runs before the history was added have none
        print(f"seed {seed}: resumed from {ckpt} at epoch {start}", flush=True)
    for ep in range(start, epochs):
        net.train()
        terms, seen_p, seen_y = [], [], []
        for idx in torch.randperm(len(lab)).split(2):
            if len(idx) < 2:                                         # BatchNorm needs more than one sample
                continue
            v, m = augment(vol[idx].to(DEVICE), mask[idx].to(DEVICE))
            l, prob = lab[idx].to(DEVICE), torch.zeros_like(v)
            for _ in range(iters):                                   # Alg. 1: one Adam step per iteration
                seg, logit = net(v[:, None] + prob[:, None])
                focal, dice = focal_loss(logit, l, w_m, w_n), dice_loss(seg, m)
                loss = lam * focal + (1 - lam) * dice
                opt.zero_grad(), loss.backward(), opt.step()
                prob = torch.sigmoid(seg).detach()
            terms.append((focal.item(), dice.item()))                # last pass, as the network is used at test time
            seen_p.append(torch.sigmoid(logit).detach().cpu()), seen_y.append(l.cpu())
        sched.step()
        p_tr, y_tr = torch.cat(seen_p).numpy(), torch.cat(seen_y).numpy()
        row = dict(epoch=ep + 1, focal=np.mean([a for a, _ in terms]), dice_loss=np.mean([b for _, b in terms]),
                   train_auc=roc_auc_score(y_tr, p_tr), p_min=p_tr.min(), p_max=p_tr.max())
        if va is not None:
            p_va, d_va, _ = predict(net, X, va, iters)
            row.update(val_auc=roc_auc_score(y[va], p_va), val_dice=d_va.mean())
        history.append({k: float(v) for k, v in row.items()})      # plain floats: torch.load(weights_only) refuses numpy
        print(f"seed {seed} " + " ".join(f"{k} {v:.3f}" if k != "epoch" else f"epoch {v}" for k, v in row.items()),
              flush=True)
        if ep % 10 == 9 or ep + 1 == epochs:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1,
                            history=history), ckpt)
    pd.DataFrame(history).to_csv(Path(ckpt_dir) / f"{tag}_history.csv", index=False)
    probs, dices, mids = predict(net, X, te, iters)
    np.savez_compressed(Path(ckpt_dir) / f"{tag}_test.npz", prob=probs, dice=dices, seg_mid=mids,
                        vol_mid=X["vol"][te][:, :, X["vol"].shape[2] // 2],
                        mask_mid=X["mask"][te][:, :, X["vol"].shape[2] // 2])
    return probs, dices.mean()


def main(epochs, seeds=(0, 1, 2), ckpt_dir=".", val=False):
    X, tv, te = load_raw()
    y = X["y"]
    va = None
    if val:                                                          # diagnostic run: 20% of Train+Validation held out
        tv, va = train_test_split(tv, test_size=0.2, stratify=y[tv], random_state=0)
    runs = []
    for s in seeds:
        runs.append(train_eval(X, y, tv, te, epochs, s, ckpt_dir=ckpt_dir, va=va))
        np.save(Path(ckpt_dir) / f"zhou2021_probs_seed{s}{'_val' if val else ''}.npy", runs[-1][0])                  # keep each seed even if a later one fails
    rows = {f"seed {s}": {**metrics(y[te], p), "Dice": d} for s, (p, d) in zip(seeds, runs)}
    mean = np.mean([p for p, _ in runs], 0)
    rows["mean of seeds"] = {**metrics(y[te], mean), "Dice": np.mean([d for _, d in runs])}
    table = pd.DataFrame(rows).T.round(4)
    print(table.to_string())
    print("AUC 95%% CI of the seed ensemble: %.3f-%.3f" % tuple(auc_ci(y[te], mean)))
    return table


ABLATIONS = {                                                        # the full method (N=2, three scales) is main()
    "vnet": dict(lam=0.0, iters=1),                                  # segmentation only
    "cnet": dict(lam=1.0, iters=1),                                  # classification only
    "cvnet": dict(iters=1, scales=(5,)),                             # multi-task, one scale, no refinement
    "cmsvnet": dict(iters=1),                                        # multi-task, multi-scale, no refinement
    "lam07": dict(lam=0.7),                                          # not in the paper: more weight on classification
}


def ablations(epochs, seed=0, ckpt_dir=".", names=tuple(ABLATIONS)):
    X, tv, te = load_raw()
    y, rows = X["y"], {}
    for n in names:
        p, d = train_eval(X, y, tv, te, epochs, seed, ckpt_dir=ckpt_dir, name=n, **ABLATIONS[n])
        rows[n] = {**metrics(y[te], p), "Dice": d}
        lam = ABLATIONS[n].get("lam")
        if lam == 0:                                                 # the untrained task's numbers mean nothing
            rows[n] = dict.fromkeys(rows[n], np.nan) | {"Dice": d}
        elif lam == 1:
            rows[n]["Dice"] = np.nan
        pd.DataFrame(rows).T.round(4).to_csv(Path(ckpt_dir) / "zhou2021_ablations.csv")       # keep finished ones
    return pd.DataFrame(rows).T.round(4)
