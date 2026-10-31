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

The paper's fixed 100 epochs overfit: training accuracy passes 99% by epoch 2 on whole slices, so `early_stopping=True`, the
default, is our own addition. It fits Train only and picks the epoch by soft-voting AUC on the official Validation split,
with the learning rate halved on a plateau. `early_stopping=False` gives the paper's schedule (`results/yu2024/yu2024_*`).
Each epoch saves a checkpoint that a restarted run resumes from; a run past `deadline` stops cleanly and resumes later.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from tdsc_abus2023_pytorch import TDSC, DataSplits
from torch.nn import functional as F

from abus_classification.data.lesion_cache import ROOT, TDSC_PATH
from abus_classification.papers.yu2024.model import YuResNet
from abus_classification.protocol import DEVICE, auc_ci, folds, metrics, run_on_gpus

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


def case_slices(split, i):
    vol, mask, label, _ = TDSC(path=TDSC_PATH, split=split, download=False)[i]
    return (*tumor_slices(np.asarray(vol), mask), int(label == 0))    # label 0 = malignant


def build_cache():
    """All CPU cores, one volume per task."""
    splits = (DataSplits.TRAIN, DataSplits.VALIDATION, DataSplits.TEST)
    tasks = [(code, split, i) for code, split in enumerate(splits) for i in range(len(TDSC(path=TDSC_PATH, split=split, download=False)))]
    done = Parallel(n_jobs=os.cpu_count())(delayed(case_slices)(split, i) for _, split, i in tasks)
    whole, crop, voxels, y = zip(*done)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, whole=np.concatenate(whole), crop=np.concatenate(crop), voxels=np.concatenate(voxels),
                        case=np.repeat(np.arange(len(done)), [len(v) for v in voxels]), y=np.array(y), split=np.array([t[0] for t in tasks]))


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


def log(out_dir, text):
    """To stdout and to a file in the kernel's output, stamped; spawned workers may not reach the kernel's own log."""
    line = f"{time.strftime('%H:%M:%S')} {text}"
    print(line, flush=True)
    with open(Path(out_dir) / "yu2024_progress.log", "a") as f:
        f.write(line + "\n")


def save_atomic(state, path):
    """A stop during the write leaves the previous checkpoint intact."""
    torch.save(state, f"{path}.tmp")
    os.replace(f"{path}.tmp", path)


def train_run(data, mode, seed, fit, test, name, out_dir=".", epochs=100, batch=32, lr=1e-4, make=None, val=None, patience=10, deadline=None):
    """Train on the slices of cases `fit`, return p(malignant) for the slices of cases `test`.

    With `val`, keep the epoch with the best soft-voting AUC there, halve lr on a 3-epoch plateau and stop after `patience`.
    A finished run is loaded and a stopped one resumes. Past `deadline` (time.time()) it returns None with its checkpoint saved.
    """
    out = Path(out_dir) / f"yu2024_{name}.npz"
    if out.exists():
        return np.load(out)["prob"]
    torch.manual_seed(seed), np.random.seed(seed)
    images, case = data[MODES[mode]["images"]], data["case"]
    idx = np.flatnonzero(np.isin(case, fit))
    x, target = torch.as_tensor(images[idx], device=DEVICE), torch.as_tensor(data["y"][case[idx]], device=DEVICE)
    net = (make or (lambda: YuResNet(MODES[mode]["pretrained"])))().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=3)
    if val is not None:
        val_slices = np.isin(case, val)
        val_x, val_case, val_voxels = torch.as_tensor(images[val_slices]), case[val_slices], data["voxels"][val_slices]
    state = dict(epoch=0, best=None, best_auc=-1.0, best_epoch=0, since=0, history=[])
    checkpoint = out.with_suffix(".ckpt")
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        net.load_state_dict(saved["net"]), opt.load_state_dict(saved["opt"]), plateau.load_state_dict(saved["plateau"])
        np.random.set_state(saved["numpy"]), state.update(saved["state"])
        log(out_dir, f"{name} resumed after epoch {state['epoch']}")
    log(out_dir, f"{name} start on {DEVICE}:{torch.cuda.current_device() if DEVICE == 'cuda' else 0}, {len(idx)} slices")
    started = time.time()
    for epoch in range(state["epoch"] + 1, epochs + 1):
        if deadline is not None and time.time() > deadline:
            log(out_dir, f"{name} stopped at the deadline after epoch {state['epoch']}")
            return None
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
        h = dict(epoch=epoch, loss=float(np.mean(losses)), acc=hits / len(idx), lr=opt.param_groups[0]["lr"])
        stop = False
        if val is not None:
            cases, _, soft = votes(predict(net, val_x), val_voxels, val_case)
            h["val_auc"] = roc_auc_score(data["y"][cases], soft)
            plateau.step(h["val_auc"])
            state["since"] += 1
            if h["val_auc"] > state["best_auc"]:
                state.update(best={k: v.cpu().clone() for k, v in net.state_dict().items()}, best_auc=h["val_auc"], best_epoch=epoch, since=0)
            stop = state["since"] >= patience
        state["history"].append(h)
        state["epoch"] = epoch
        save_atomic(dict(net=net.state_dict(), opt=opt.state_dict(), plateau=plateau.state_dict(), numpy=np.random.get_state(), state=state), checkpoint)
        pd.DataFrame(state["history"]).to_csv(out.with_name(out.stem + "_history.csv"), index=False)
        log(out_dir, f"{name} epoch {epoch} loss {h['loss']:.3f} acc {h['acc']:.3f} lr {h['lr']:.1e} val AUC {h.get('val_auc', float('nan')):.3f} "
                     f"best {state['best_auc']:.3f} at {state['best_epoch']} {time.time() - started:.0f} s since start")
        if stop:
            log(out_dir, f"{name} early stop at epoch {epoch}")
            break
    if state["best"] is not None:
        net.load_state_dict(state["best"])
    prob = predict(net, torch.as_tensor(images[np.isin(case, test)]))
    torch.save(net.state_dict(), out.with_suffix(".pt"))
    np.savez_compressed(out, prob=prob, best_epoch=state["best_epoch"], val_auc=state["best_auc"])
    checkpoint.unlink(missing_ok=True)
    log(out_dir, f"{name} done, best epoch {state['best_epoch']}, val AUC {state['best_auc']:.3f}")
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


def plan(data, modes, seeds, early_stopping=True):
    """Every training: three seeds per mode on the official split, then the paper's five folds on Train for pretrained modes."""
    y, split = data["y"], data["split"]
    train, validation, test = [np.flatnonzero(split == k) for k in range(3)]
    tag, val = ("es_", validation) if early_stopping else ("", None)
    fit = train if early_stopping else np.r_[train, validation]
    runs = []
    for mode in modes:
        runs += [dict(mode=mode, seed=s, fit=fit, test=test, name=f"{tag}{mode}_seed{s}", val=val, key=f"{mode} seed {s}") for s in seeds]
        if MODES[mode]["pretrained"]:
            runs += [dict(mode=mode, seed=0, fit=a, test=b, name=f"{tag}cv_{mode}_seed0_fold{k}", val=val, key=f"{mode} fold {k}")
                     for k, (a, b) in enumerate(folds(y, train, 0))]
    return runs


def run_job(job):
    """One training in a GPU worker. The cache is read per process, and only the images it needs stay on the GPU."""
    data = dict(np.load(CACHE))
    train_run(data, job["mode"], job["seed"], job["fit"], job["test"], job["name"], job["out_dir"], job["epochs"], val=job["val"], deadline=job["deadline"])


def main(modes=tuple(MODES), seeds=(0, 1, 2), out_dir=".", epochs=100, early_stopping=True, hours=None):
    """`hours` limits the run, so that a kernel ends cleanly and keeps its output before the platform stops it."""
    deadline = None if hours is None else time.time() + hours * 3600
    if not CACHE.exists():
        log(out_dir, "building the cache")
        build_cache()
    data = dict(np.load(CACHE))
    log(out_dir, f"{(data['split'][data['case']] == 0).sum()} slices with tumor on Train (paper: 2028), {torch.cuda.device_count()} GPUs, {os.cpu_count()} CPUs")
    runs = plan(data, modes, seeds, early_stopping)
    run_on_gpus(run_job, [{**r, "out_dir": out_dir, "epochs": epochs, "deadline": deadline} for r in runs])
    tag, rows, fold_rows, soft = "es_" if early_stopping else "", {}, {}, {}
    for r in runs:
        path = Path(out_dir) / f"yu2024_{r['name']}.npz"
        if not path.exists():
            log(out_dir, f"{r['name']} not finished")
            continue
        result, y, _, s = evaluate(data, np.load(path)["prob"], r["test"])
        if "fold" in r["key"]:
            fold_rows.setdefault(r["mode"], []).append(result)
        else:
            rows.update({f"{r['mode']} {kind} seed {r['seed']}": v for kind, v in result.items()})
            soft.setdefault(r["mode"], []).append(s)
    for mode, s in soft.items():
        y = data["y"][data["split"] == 2]
        log(out_dir, f"{mode} soft voting, mean of seeds, AUC 95% CI {auc_ci(y, np.mean(s, 0))}")
        rows[f"{mode} soft mean of seeds"] = row(y, np.mean(s, 0))
    for mode, results in fold_rows.items():
        rows.update({f"{mode} {kind} cv": pd.DataFrame([f[kind] for f in results]).mean().to_dict() for kind in results[0]})
    table = pd.DataFrame(rows).T.round(4)
    table.to_csv(Path(out_dir) / f"yu2024_{tag}results.csv")
    print(table.to_string())
    return table
