"""Yu et al., Applied Sciences 14(24):11611 (2024): a 2D ResNet34 on tumor slices, then hard or soft voting per volume.

    python -m abus_classification.papers.yu2024         # builds data/yu2024.npz from TDSC_PATH; train on a GPU

Paper details followed: every slice with tumor (axis 2, 0.476 mm apart, 2028 on Train) is one sample with its volume's
label; ResNet34; 100 epochs; Eq. 1 soft voting, the voxel-weighted mean of slice probabilities; Eq. 3 hard voting, the
majority of slices; five-fold CV on Train (100 volumes), split by volume, as a second evaluation (`cv`).
Deviations: the main evaluation fits Train + Validation and reports on the official Test split (paper: CV only); only
ResNet34 (paper also MedViT, HiFuse, MedMamba and 3D networks, all below it in its Table 1 or 2).
Not given and assumed: ImageNet weights (`resnet34_scratch` runs without); whole slices as Fig. 2 draws them (`resnet34_crop`
uses the slice's lesion box, as its discussion of "smaller slices" may mean); 224 x 224 ignoring aspect ratio; Adam, lr 1e-4,
batch 32, cross-entropy, no augmentation; the last epoch is used. A hard-voting tie counts as malignant (paper: undefined).
Voting weights come from the ground-truth masks, as in the paper, so a real system would need a segmentation first.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tdsc_abus2023_pytorch import TDSC, DataSplits
from torch.nn import functional as F

from abus_classification.data.lesion_cache import ROOT, TDSC_PATH
from abus_classification.papers.yu2024.model import YuResNet
from abus_classification.protocol import DEVICE, auc_ci, folds, metrics

CACHE, SIZE = ROOT / "data" / "yu2024.npz", 224
MODES = {"resnet34": dict(images="whole", pretrained=True),
         "resnet34_crop": dict(images="crop", pretrained=True),
         "resnet34_scratch": dict(images="whole", pretrained=False)}


def resize(plane, size=SIZE):
    x = torch.as_tensor(np.asarray(plane, np.float32))[None, None]
    return F.interpolate(x, (size, size), mode="bilinear", align_corners=False, antialias=True)[0, 0].clamp(0, 255).round().numpy().astype(np.uint8)


def tumor_slices(vol, mask, size=SIZE):
    """Slices with tumor: whole (n, size, size), cropped to the slice's lesion box, and tumor voxels per slice (n)."""
    mask = np.asarray(mask) > 0
    whole, crop, voxels = [], [], []
    for z in np.flatnonzero(mask.any((0, 1))):
        points = np.argwhere(mask[:, :, z])
        (a, b), (c, d) = points.min(0), points.max(0) + 1
        whole.append(resize(vol[:, :, z], size))
        crop.append(resize(vol[a:c, b:d, z], size))
        voxels.append(len(points))
    return np.stack(whole), np.stack(crop), np.array(voxels)


def build_cache():
    rows, case = {k: [] for k in ("whole", "crop", "voxels", "case")}, 0
    y, split_of = [], []
    for code, split in enumerate((DataSplits.TRAIN, DataSplits.VALIDATION, DataSplits.TEST)):
        ds = TDSC(path=TDSC_PATH, split=split, download=False)
        for i in range(len(ds)):
            vol, mask, label, _ = ds[i]
            whole, crop, voxels = tumor_slices(np.asarray(vol), mask)
            for key, v in zip(rows, (whole, crop, voxels, np.full(len(voxels), case))):
                rows[key].append(v)
            y.append(int(label == 0)), split_of.append(code)          # label 0 = malignant
            case += 1
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **{k: np.concatenate(v) for k, v in rows.items()}, y=np.array(y), split=np.array(split_of))


def votes(p, voxels, case):
    """Per volume, for slice probabilities p(malignant): cases, hard vote (share of malignant slices), soft vote (Eq. 1)."""
    cases = np.unique(case)
    hard = np.array([(p[case == c] > 0.5).mean() for c in cases])
    soft = np.array([np.average(p[case == c], weights=voxels[case == c]) for c in cases])
    return cases, hard, soft


def predict(net, x, batch=128):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(F.softmax(net(x[i:i + batch, None].to(DEVICE).float() / 255), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train_run(data, mode, seed, fit, test, name, out_dir=".", epochs=100, batch=32, lr=1e-4, make=None):
    """Train on the slices of cases `fit`, return p(malignant) for the slices of cases `test`. A finished run is loaded."""
    out = Path(out_dir) / f"yu2024_{name}.npz"
    if out.exists():
        return np.load(out)["prob"]
    torch.manual_seed(seed), np.random.seed(seed)
    images, case = data[MODES[mode]["images"]], data["case"]
    idx = np.flatnonzero(np.isin(case, fit))
    x, target = torch.as_tensor(images[idx], device=DEVICE), torch.as_tensor(data["y"][case[idx]], device=DEVICE)
    net = (make or (lambda: YuResNet(MODES[mode]["pretrained"])))().to(DEVICE)
    opt, history = torch.optim.Adam(net.parameters(), lr=lr), []
    for epoch in range(1, epochs + 1):
        net.train()
        order, losses, hits = np.random.permutation(len(idx)), [], 0
        for start in range(0, len(order), batch):
            pick = order[start:start + batch]
            if len(pick) < 2:                                         # BatchNorm needs more than one sample
                continue
            logits = net(x[pick, None].float() / 255)
            loss = F.cross_entropy(logits, target[pick])
            opt.zero_grad(), loss.backward(), opt.step()
            losses.append(loss.item())
            hits += (logits.argmax(1) == target[pick]).sum().item()
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), acc=hits / len(idx)))
        print(f"{name} epoch {epoch} loss {history[-1]['loss']:.3f} acc {history[-1]['acc']:.3f}", flush=True)
    prob = predict(net, torch.as_tensor(images[np.isin(case, test)]))
    pd.DataFrame(history).to_csv(out.with_name(out.stem + "_history.csv"), index=False)
    torch.save(net.state_dict(), out.with_suffix(".pt"))
    np.savez_compressed(out, prob=prob)
    return prob


def row(y, p):
    """The protocol's metrics, the paper's score (Eq. 4) and (Sens + Spec) / 2: its voting rows report AUC = score, as from labels."""
    m = metrics(y, p)
    return {**m, "Score": (m["Acc"] + m["AUC"]) / 2, "Balanced": (m["Sens"] + m["Spec"]) / 2}


def evaluate(data, prob, test):
    """Rows for slices, hard voting and soft voting. Slices take their volume's label."""
    keep = np.isin(data["case"], test)
    case, voxels = data["case"][keep], data["voxels"][keep]
    cases, hard, soft = votes(prob, voxels, case)
    y = data["y"][cases]
    return {"slices": row(data["y"][case], prob), "hard": row(y, hard), "soft": row(y, soft)}, y, hard, soft


def cv(data, mode, out_dir=".", epochs=100, seed=0):
    """The paper's protocol: five folds on Train, the mean of the folds' metrics."""
    y, train = data["y"], np.flatnonzero(data["split"] == 0)
    results = []
    for k, (fit, test) in enumerate(folds(y, train, seed)):
        prob = train_run(data, mode, seed, fit, test, f"cv_{mode}_seed{seed}_fold{k}", out_dir, epochs)
        results.append(evaluate(data, prob, test)[0])
    return {f"{mode} {kind} cv": pd.DataFrame([r[kind] for r in results]).mean().to_dict() for kind in results[0]}


def main(modes=tuple(MODES), seeds=(0, 1, 2), out_dir=".", epochs=100):
    if not CACHE.exists():
        build_cache()
    data = dict(np.load(CACHE))
    fit, test = np.flatnonzero(data["split"] != 2), np.flatnonzero(data["split"] == 2)
    print(f"{(data['split'][data['case']] == 0).sum()} slices with tumor on Train (paper: 2028)", flush=True)
    rows = {}
    for mode in modes:
        soft = []
        for seed in seeds:
            prob = train_run(data, mode, seed, fit, test, f"{mode}_seed{seed}", out_dir, epochs)
            result, y, _, s = evaluate(data, prob, test)
            rows.update({f"{mode} {kind} seed {seed}": v for kind, v in result.items()})
            soft.append(s)
            pd.DataFrame(rows).T.round(4).to_csv(Path(out_dir) / "yu2024_results.csv")      # keep finished runs
        print(mode, "soft voting, mean of seeds, AUC 95%% CI %.3f-%.3f" % tuple(auc_ci(y, np.mean(soft, 0))), flush=True)
        rows[f"{mode} soft mean of seeds"] = row(y, np.mean(soft, 0))
        if MODES[mode]["pretrained"]:
            rows.update(cv(data, mode, out_dir, epochs))
        table = pd.DataFrame(rows).T.round(4)
        table.to_csv(Path(out_dir) / "yu2024_results.csv")
    print(table.to_string())
    return table
