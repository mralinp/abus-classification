"""Hejduk et al., Eur. Radiol. 32:4868-4878 (2022): three step-wise dCNNs and a sliding window over ABUS slices.

    python -m abus_classification.papers.hejduk2022     # builds data/hejduk2022.npz from TDSC_PATH; train on a GPU

Paper details followed: slices at 600 x 600, crops of 151 x 181 pixels; model 1 background vs tissue, model 2 normal
tissue vs lesion, model 3 suspicious vs not; 13 conv, 5 max pool, 2 dense, 50% dropout, softmax; SGD, Nesterov 0.9, lr 1e-6,
batch 16, binary cross-entropy, class weights; 120, 120 and 160 epochs; rotation, shifts, shear, zoom and horizontal flip;
window centre moved by 10 pixels (a 60 x 60 map per slice); a model only sees windows the previous one accepted; a volume
takes the highest suspicion score found in it.
Deviations: labels are malignant vs benign from TDSC (paper: BI-RADS 4/5 vs 2/3 from reports, with no BI-RADS in TDSC);
crops are centred on the ground-truth mask (paper: drawn by a radiologist), five per lesion (paper: about four per patient);
fixed epochs on Train + Validation, reported on the official Test split (paper: own split, 128 images and 22 volumes).
The Table 1 image counts are drawn afresh each epoch (paper: fixed augmented copies), which keeps the paper's step count.
Not given and assumed: 181 is the height along depth; VGG-16 filters and 4096-unit dense layers; pixels scaled to [0, 1];
augmentation ranges; Xavier initialisation (TensorFlow's default); tissue is a pixel >= 10, background a crop under 10%
tissue, normal a crop over 90% tissue that misses the lesion. Boxes for the challenge benchmark are our own addition.

`early_stopping=True`, the default, is our addition to the paper's fixed epochs, so that overfitting shows early: models fit
Train, the epoch is picked by crop AUC on the official Validation split, and lr halves on a plateau. Models train one per
GPU. Training saves a checkpoint each epoch and the scan each volume; a restarted run resumes, and `hours` ends it cleanly.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy import ndimage
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from tdsc_abus2023_pytorch import TDSC, DataSplits
from torch.nn import functional as F

from abus_classification.data.lesion_cache import ROOT, TDSC_PATH
from abus_classification.papers.hejduk2022.model import HEIGHT, WIDTH, HejdukCNN
from abus_classification.protocol import DEVICE, auc_ci, metrics, run_on_gpus, save_atomic
from abus_classification.protocol import log as protocol_log

CACHE, SIZE = ROOT / "data" / "hejduk2022.npz", 600
BACKGROUND, NORMAL, LESION = 0, 1, 2
# kinds of crop per class, images per epoch (Table 1) and epochs
MODELS = {1: dict(classes=((BACKGROUND,), (NORMAL, LESION)), images=2560, epochs=120),
          2: dict(classes=((NORMAL,), (LESION,)), images=6241, epochs=120),
          3: dict(classes=None, images=7164, epochs=160)}


def to_image(plane, mode="bilinear"):
    """A slice (axis 0, depth) as a (depth, width) 600 x 600 tensor."""
    x = torch.as_tensor(np.asarray(plane, np.float32).T)[None, None]
    kw = dict(mode=mode, align_corners=False, antialias=True) if mode == "bilinear" else dict(mode=mode)
    return F.interpolate(x, (SIZE, SIZE), **kw)[0, 0].round().clamp(0, 255).to(torch.uint8)


def pad(image):
    """Zeros around an image, so that pad(image)[r:r + HEIGHT, c:c + WIDTH] is the window centred on (r, c)."""
    return F.pad(image, (WIDTH // 2, WIDTH // 2, HEIGHT // 2, HEIGHT // 2))


def windows(image, stride=10):
    """Every window of a slice, (n * n, HEIGHT, WIDTH), centres stride / 2, 3 * stride / 2, ...; n = 60 at stride 10."""
    grid = torch.arange(stride // 2, SIZE, stride)
    w = pad(image).unfold(0, HEIGHT, 1).unfold(1, WIDTH, 1)
    return w[grid][:, grid].flatten(0, 1).clone()


def window_share(image, stride=10):
    """The share of each window's pixels that are non-zero, as an n x n array."""
    x = pad(image.float())[None, None]
    return F.avg_pool2d(x, (HEIGHT, WIDTH), 1)[0, 0, stride // 2::stride, stride // 2::stride].numpy()


def volume_crops(vol, mask, rng, per_kind=5, tissue=10):
    """Lesion, normal and background crops of a volume. Returns crops (n, HEIGHT, WIDTH) and kinds (n)."""
    vol, mask = np.asarray(vol), np.asarray(mask) > 0
    crops, kinds = [], []
    lesion_slices = np.flatnonzero(mask.any((0, 1)))
    for z in lesion_slices[np.linspace(0, len(lesion_slices) - 1, per_kind).round().astype(int)]:
        points = np.argwhere(mask[:, :, z])
        col, row = (points.min(0) + points.max(0)) / 2 * SIZE / np.array(mask.shape[:2])
        r, c = int(row), int(col)
        crops.append(pad(to_image(vol[:, :, z]))[r:r + HEIGHT, c:c + WIDTH].numpy())
        kinds.append(LESION)
    candidates = {NORMAL: [], BACKGROUND: []}
    for z in np.linspace(0, vol.shape[2] - 1, 20).round().astype(int):
        image = to_image(vol[:, :, z])
        share = window_share((image >= tissue).to(torch.uint8))
        hit = window_share(to_image(mask[:, :, z], "nearest"))
        candidates[NORMAL] += [(image, r, c) for r, c in np.argwhere((share > 0.9) & (hit == 0))]
        candidates[BACKGROUND] += [(image, r, c) for r, c in np.argwhere(share < 0.1)]
    for kind, found in candidates.items():
        for i in rng.permutation(len(found))[:per_kind]:
            image, r, c = found[i]
            r, c = r * 10 + 5, c * 10 + 5
            crops.append(pad(image)[r:r + HEIGHT, c:c + WIDTH].numpy())
            kinds.append(kind)
    return np.stack(crops), np.array(kinds)


def case_crops(code, split, i, case):
    """Test volumes only give lesion crops: their windows are evaluated in place."""
    vol, mask, label, _ = TDSC(path=TDSC_PATH, split=split, download=False)[i]
    crops, kinds = volume_crops(vol, mask, np.random.default_rng(case))
    if code == 2:
        crops, kinds = crops[kinds == LESION], kinds[kinds == LESION]
    n = len(kinds)
    return crops, kinds, np.full(n, int(label == 0)), np.full(n, code), np.full(n, case)      # label 0 = malignant


def build_cache():
    """All CPU cores, one volume per task."""
    splits = enumerate((DataSplits.TRAIN, DataSplits.VALIDATION, DataSplits.TEST))
    tasks = [(code, split, i) for code, split in splits for i in range(len(TDSC(path=TDSC_PATH, split=split, download=False)))]
    done = Parallel(n_jobs=os.cpu_count())(delayed(case_crops)(code, split, i, case) for case, (code, split, i) in enumerate(tasks))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **{k: np.concatenate(v) for k, v in zip(("crops", "kind", "y", "split", "case"), zip(*done))})


def training_set(data, model, splits=(0, 1)):
    """Indices of the crops a model trains on, and their targets; by default from Train + Validation."""
    fit, kind = np.isin(data["split"], splits), data["kind"]
    classes = MODELS[model]["classes"]
    if classes is None:
        idx = np.flatnonzero(fit & (kind == LESION))
        return idx, data["y"][idx]
    negative, positive = np.isin(kind, classes[0]), np.isin(kind, classes[1])
    idx = np.flatnonzero(fit & (negative | positive))
    return idx, positive[idx].astype(int)


def class_weights(y):
    """TensorFlow's balanced class_weight: n / (2 * n_class)."""
    return len(y) / (2 * np.bincount(y, minlength=2))


def affine(rotate, shear, zoom, dx, dy, flip, aspect=WIDTH / HEIGHT):
    """Sampling matrix (n, 2, 3) for grid_sample; angles in radians, shifts in normalised units, flip in {1, -1}."""
    a, b = rotate + shear, rotate
    theta = torch.zeros(len(rotate), 2, 3).to(rotate)
    theta[:, 0, 0], theta[:, 0, 1] = a.cos() * zoom * flip, -b.sin() * zoom / aspect   # computed in pixels: non-square crops
    theta[:, 1, 0], theta[:, 1, 1] = a.sin() * zoom * flip * aspect, b.cos() * zoom
    theta[:, 0, 2], theta[:, 1, 2] = dx, dy
    return theta


def augment(x):
    """Rotation +-20 degrees, shifts, shear and zoom +-10%, horizontal flip, per sample; edge pixels fill the gaps."""
    n = len(x)
    u = lambda r: (torch.rand(n, device=x.device) * 2 - 1) * r
    theta = affine(u(np.pi / 9), u(np.pi / 18), 1 + u(0.1), u(0.2), u(0.2), torch.rand(n, device=x.device).round() * 2 - 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


def predict(net, x, batch=512):
    """p(class 1) for uint8 windows (n, HEIGHT, WIDTH)."""
    out = torch.zeros(len(x))
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", enabled=DEVICE == "cuda"):
        for i in range(0, len(x), batch):
            logits = net(x[i:i + batch, None].to(DEVICE).float() / 255)
            out[i:i + batch] = F.softmax(logits.float(), 1)[:, 1].cpu()
    return out


def stepwise(nets, x):
    """Probabilities (3, n): a model only sees the windows the previous one accepted; the others stay at zero."""
    maps, keep = torch.zeros(3, len(x)), torch.arange(len(x))
    for k, net in enumerate(nets):
        if len(keep) == 0:
            break
        maps[k, keep] = predict(net, x[keep])
        keep = keep[maps[k, keep] >= 0.5]
    return maps


def volume_maps(nets, vol, stride=10, slice_step=1):
    """Probability maps of a volume: (slices, 3, n, n), the paper's 60 x 60 x 3 x 330 at stride 10."""
    vol = np.asarray(vol)
    n = len(range(stride // 2, SIZE, stride))
    return torch.stack([stepwise(nets, windows(to_image(vol[:, :, z]), stride)).reshape(3, n, n)
                        for z in range(0, vol.shape[2], slice_step)]).numpy()


def detect(maps, shape, stride=10, slice_step=1):
    """Lesions as (box, score): connected windows models 1 and 2 accepted. Boxes as `challenge.mask_box`, array axes 2, 1, 0."""
    found = maps[:, 1] >= 0.5
    labels, count = ndimage.label(found)
    boxes = []
    for k, region in enumerate(ndimage.find_objects(labels), 1):
        lo = [s.start for s in region]
        hi = [s.stop for s in region]                                 # one cell past the last centre
        scale_y, scale_z = stride * shape[1] / SIZE, stride * shape[0] / SIZE
        x, y, z = (lo[0] * slice_step, hi[0] * slice_step), (lo[1] * scale_y, hi[1] * scale_y), (lo[2] * scale_z, hi[2] * scale_z)
        box = tuple((a + b) / 2 for a, b in (x, y, z)) + tuple(b - a for a, b in (x, y, z))
        boxes.append((box, float(maps[:, 1][labels == k].max())))
    return boxes


def log(out_dir, text):
    protocol_log(Path(out_dir) / "hejduk2022_progress.log", text)


def train_model(data, model, seed, out_dir=".", lr=1e-6, batch=16, epochs=None, make=HejdukCNN, early_stopping=False, patience=10, deadline=None):
    """Train one of the three models. A finished model is loaded and a stopped one resumes; past `deadline` it returns None."""
    tag = ("_es" if early_stopping else "") + ("" if lr == 1e-6 else f"_lr{lr:g}")
    path = Path(out_dir) / f"hejduk2022_model{model}{tag}_seed{seed}.pt"
    net = make().to(DEVICE)
    if path.exists():
        net.load_state_dict(torch.load(path, map_location=DEVICE))
        return net
    torch.manual_seed(seed), np.random.seed(seed)
    idx, y = training_set(data, model, (0,) if early_stopping else (0, 1))
    crops, target = torch.as_tensor(data["crops"][idx], device=DEVICE), torch.as_tensor(y, device=DEVICE)
    weight = torch.as_tensor(class_weights(y), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=3)
    if early_stopping:
        val_idx, val_y = training_set(data, model, (1,))
        val_crops = torch.as_tensor(data["crops"][val_idx])
    state = dict(epoch=0, best=None, best_auc=-1.0, best_epoch=0, since=0, history=[])
    checkpoint, name = path.with_suffix(".ckpt"), path.stem
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        net.load_state_dict(saved["net"]), opt.load_state_dict(saved["opt"]), plateau.load_state_dict(saved["plateau"])
        np.random.set_state(saved["numpy"]), state.update(saved["state"])
        log(out_dir, f"{name} resumed after epoch {state['epoch']}")
    log(out_dir, f"{name} start on {DEVICE}:{torch.cuda.current_device() if DEVICE == 'cuda' else 0}, {len(idx)} crops")
    images, started = MODELS[model]["images"], time.time()
    for epoch in range(state["epoch"] + 1, (epochs or MODELS[model]["epochs"]) + 1):
        if deadline is not None and time.time() > deadline:
            log(out_dir, f"{name} stopped at the deadline after epoch {state['epoch']}")
            return None
        net.train()
        order, losses, hits = np.random.randint(0, len(idx), images), [], 0
        for start in range(0, images, batch):
            pick = order[start:start + batch]
            logits = net(augment(crops[pick, None].float() / 255))
            loss = (F.cross_entropy(logits, target[pick], reduction="none") * weight[target[pick]]).mean()
            opt.zero_grad(), loss.backward(), opt.step()
            losses.append(loss.item())
            hits += (logits.argmax(1) == target[pick]).sum().item()
        h = dict(epoch=epoch, loss=float(np.mean(losses)), acc=hits / images, lr=opt.param_groups[0]["lr"])
        stop = False
        if early_stopping:
            h["val_auc"] = roc_auc_score(val_y, predict(net, val_crops).numpy())
            plateau.step(h["val_auc"])
            state["since"] += 1
            if h["val_auc"] > state["best_auc"]:
                state.update(best={k: v.cpu().clone() for k, v in net.state_dict().items()}, best_auc=h["val_auc"], best_epoch=epoch, since=0)
            stop = state["since"] >= patience
        state["history"].append(h)
        state["epoch"] = epoch
        save_atomic(dict(net=net.state_dict(), opt=opt.state_dict(), plateau=plateau.state_dict(), numpy=np.random.get_state(), state=state), checkpoint)
        pd.DataFrame(state["history"]).to_csv(path.with_name(path.stem + "_history.csv"), index=False)
        log(out_dir, f"{name} epoch {epoch} loss {h['loss']:.3f} acc {h['acc']:.3f} lr {h['lr']:.1e} val AUC {h.get('val_auc', float('nan')):.3f} "
                     f"best {state['best_auc']:.3f} at {state['best_epoch']} {time.time() - started:.0f} s since start")
        if stop:
            log(out_dir, f"{name} early stop at epoch {epoch}")
            break
    if state["best"] is not None:
        net.load_state_dict(state["best"])
    torch.save(net.state_dict(), path)
    checkpoint.unlink(missing_ok=True)
    log(out_dir, f"{name} done, best epoch {state['best_epoch']}, val AUC {state['best_auc']:.3f}")
    return net


def evaluate(data, nets, seed, out_dir=".", lr=1e-6, stride=10, slice_step=1, tag="", deadline=None):
    """Test results of one seed: per lesion crop, then per volume. A finished seed is loaded, and each scanned volume is kept."""
    tag = tag + ("" if lr == 1e-6 else f"_lr{lr:g}")
    out = Path(out_dir) / f"hejduk2022{tag}_seed{seed}_test.npz"
    if out.exists():
        return dict(np.load(out))
    te = np.flatnonzero((data["split"] == 2) & (data["kind"] == LESION))
    image = predict(nets[2], torch.as_tensor(data["crops"][te])).numpy()
    ds = TDSC(path=TDSC_PATH, split=DataSplits.TEST, download=False)
    volume, y, maps, boxes, parts = [], [], {}, [], []
    for i in range(len(ds)):
        part = out.with_name(f"{out.stem}_volume{i}.npz")
        parts.append(part)
        if not part.exists():
            if deadline is not None and time.time() > deadline:
                log(out_dir, f"seed {seed} scan stopped at the deadline before volume {i}")
                return None
            vol, _, label, _ = ds[i]
            m = volume_maps(nets, vol, stride, slice_step)
            found = [(i, *box, score) for box, score in detect(m, np.shape(vol), stride, slice_step)]
            with open(f"{part}.tmp", "wb") as f:
                np.savez_compressed(f, map=(m * 255).astype(np.uint8), score=m[:, 2].max(), y=int(label == 0), boxes=np.array(found, float).reshape(-1, 8))
            os.replace(f"{part}.tmp", part)
            log(out_dir, f"seed {seed} volume {i} score {m[:, 2].max():.3f} found {int((m[:, 1] >= 0.5).sum())} windows")
        saved = np.load(part)
        volume.append(saved["score"]), y.append(saved["y"]), boxes.append(saved["boxes"])
        maps[f"map_{i}"] = saved["map"]
    result = dict(image=image, image_y=data["y"][te], image_case=data["case"][te], volume=np.array(volume), y=np.array(y),
                  boxes=np.concatenate(boxes), stride=stride, slice_step=slice_step, **maps)
    np.savez_compressed(out, **result)
    for part in parts:
        part.unlink()
    return result


def train_job(job):
    data = dict(np.load(CACHE))
    train_model(data, job["model"], job["seed"], job["out_dir"], job["lr"], early_stopping=job["early_stopping"], deadline=job["deadline"])


def scan_job(job):
    """One seed's test scan, once its three models are finished."""
    data = dict(np.load(CACHE))
    tag = "_es" if job["early_stopping"] else ""
    suffix = tag + ("" if job["lr"] == 1e-6 else f"_lr{job['lr']:g}")
    if not all((Path(job["out_dir"]) / f"hejduk2022_model{m}{suffix}_seed{job['seed']}.pt").exists() for m in MODELS):
        log(job["out_dir"], f"seed {job['seed']} not scanned, its models are not finished")
        return
    nets = [train_model(data, m, job["seed"], job["out_dir"], job["lr"], early_stopping=job["early_stopping"]) for m in MODELS]
    evaluate(data, nets, job["seed"], job["out_dir"], job["lr"], job["stride"], job["slice_step"], tag, job["deadline"])


def row(y, p):
    return {**metrics(y, p), "Kappa": cohen_kappa_score(y, p >= 0.5)}


def main(seeds=(0, 1, 2), out_dir=".", lr=1e-6, stride=10, slice_step=1, early_stopping=True, hours=None):
    """`hours` limits the run, so that a kernel ends cleanly and keeps its output before the platform stops it."""
    common = dict(out_dir=out_dir, lr=lr, early_stopping=early_stopping, deadline=None if hours is None else time.time() + hours * 3600)
    if not CACHE.exists():
        log(out_dir, "building the cache")
        build_cache()
    log(out_dir, f"cache ready, {torch.cuda.device_count()} GPUs, {os.cpu_count()} CPUs")
    run_on_gpus(train_job, [dict(model=m, seed=s, **common) for s in seeds for m in MODELS])
    run_on_gpus(scan_job, [dict(seed=s, stride=stride, slice_step=slice_step, **common) for s in seeds])
    data = dict(np.load(CACHE))
    tag = ("_es" if early_stopping else "") + ("" if lr == 1e-6 else f"_lr{lr:g}")
    rows, runs = {}, []
    for seed in seeds:
        path = Path(out_dir) / f"hejduk2022{tag}_seed{seed}_test.npz"
        if not path.exists():
            log(out_dir, f"seed {seed} not finished")
            continue
        r = dict(np.load(path))
        runs.append(r)
        rows[f"images seed {seed}"] = row(r["image_y"], r["image"])
        rows[f"volumes seed {seed}"] = row(r["y"], r["volume"])
    if runs:
        mean = np.mean([r["volume"] for r in runs], 0)
        rows["volumes mean of seeds"] = row(runs[0]["y"], mean)
        log(out_dir, f"volumes ensemble AUC 95% CI {auc_ci(runs[0]['y'], mean)}")
    table = pd.DataFrame(rows).T.round(4)
    table.to_csv(Path(out_dir) / f"hejduk2022{tag}_results.csv")
    print(table.to_string())
    return table
