"""Luo et al., TDSC-ABUS challenge (arXiv:2501.15588), Sec. 3.2.1: team T1 (SZU), the winner.

    python -m abus_classification.papers.luo2025_t1     # builds data/luo2025_t1.npz from TDSC_PATH; train on a GPU

Paper details followed: three separate models. CLSNet, a 3D ResNet with softmax; SEGNet, a U-Net; DETNet, single-stage and
anchor-free after CPMNet. Metrics are the challenge's, on the official Test split.
The paper gives nothing else, so all of this is assumed: 0.5 mm voxels for SEGNet and CLSNet, 1 mm whole volumes for DETNet;
ResNet-18 depth; a 128 x 64 x 128 patch; Adam; flips and brightness augmentation; a CenterNet-style heatmap for DETNet; the
largest connected component kept by SEGNet. T1's classifier input is not stated: ours is a patch around the detected box.

Test predictions use no ground truth. CLSNet is also scored around the true lesion centre, as a reference (`prob_gt`).
Each training fits Train, keeps the best epoch on Validation, halves the learning rate on a plateau and stops early, so it
uses 100 volumes where the shared protocol says 130. Every epoch saves a checkpoint that a restarted run resumes from; a run
past `deadline` stops cleanly, leaving its checkpoint, and a later run continues it.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from tdsc_abus2023_pytorch import TDSC, DataSplits
from torch.nn import functional as F

from abus_classification import challenge
from abus_classification.data.lesion_cache import ROOT, TDSC_PATH
from abus_classification.papers.luo2025_t1.model import CLSNet, DETNet, SEGNet
from abus_classification.protocol import DEVICE as BEST_DEVICE, auc_ci, metrics, run_on_gpus
from abus_classification.utils.spacing import TDSC_SPACING

DEVICE = "cpu" if BEST_DEVICE == "mps" else BEST_DEVICE               # MPS has no conv3d
CACHE = ROOT / "data" / "luo2025_t1.npz"
ISO, PATCH, TOP_K = 0.5, (128, 64, 128), 5
SPLITS = (DataSplits.TRAIN, DataSplits.VALIDATION, DataSplits.TEST)
TASKS = dict(det=dict(steps=25, epochs=200, patience=30, lr=1e-3), seg=dict(steps=100, epochs=150, patience=20, lr=1e-3),
             cls=dict(steps=25, epochs=200, patience=30, lr=1e-4))


def log(out_dir, text):
    line = f"{time.strftime('%H:%M:%S')} {text}"
    print(line, flush=True)
    with open(Path(out_dir) / "luo2025_t1_progress.log", "a") as f:
        f.write(line + "\n")


def box_of(mask):
    """Array-axis (centre, length) of a mask, in continuous voxel coordinates: voxel i spans [i, i + 1]."""
    points = np.argwhere(mask)
    lo, hi = points.min(0), points.max(0) + 1
    return np.r_[(lo + hi) / 2, hi - lo].astype(np.float32)


def resample(vol, mask):
    """To ISO mm with area averaging: uint8 volume, bool mask (a thin lesion keeps its voxels), original shape."""
    size = [max(1, round(n * s / ISO)) for n, s in zip(vol.shape, TDSC_SPACING)]
    area = lambda a: F.interpolate(torch.as_tensor(np.asarray(a), dtype=torch.float32)[None, None], size, mode="area")[0, 0].numpy()
    v, m = area(vol), area(np.asarray(mask) > 0)
    return v.round().astype(np.uint8), m > 0.5 if (m > 0.5).any() else m > 0


def one_case(code, split, i):
    vol, mask, label, _ = TDSC(path=TDSC_PATH, split=split, download=False)[i]
    v, m = resample(vol, mask)
    return dict(vol=v, mask=m, y=int(label == 0), split=code, source=i, shape=vol.shape, box=box_of(m))   # label 0 = malignant


def build_cache():
    tasks = [(c, s, i) for c, s in enumerate(SPLITS) for i in range(len(TDSC(path=TDSC_PATH, split=s, download=False)))]
    write_cache(Parallel(n_jobs=os.cpu_count())(delayed(one_case)(*t) for t in tasks))


def write_cache(cases, path=CACHE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.array([c[k] for c in cases]) for k in ("y", "split", "source", "shape", "box")}
    arrays.update({f"{k}{i}": c[k] for i, c in enumerate(cases) for k in ("vol", "mask")})
    np.savez(path, **arrays)


def crop(v, centre, size):
    """A `size` patch of a (..., a, b, c) tensor around `centre`, zero outside the volume."""
    lo = [int(round(float(c) - s / 2)) for c, s in zip(centre, size)]
    out = torch.zeros((*v.shape[:-3], *size), dtype=v.dtype, device=v.device)
    src = tuple(slice(max(a, 0), min(a + s, n)) for a, s, n in zip(lo, size, v.shape[-3:]))
    dst = tuple(slice(max(-a, 0), max(-a, 0) + (b.stop - b.start)) for a, b in zip(lo, src))
    out[(..., *dst)] = v[(..., *src)]
    return out


def augment(x, *others):
    """The same random flips for every tensor, and a brightness and contrast change for x. Shapes: (n, a, b, c)."""
    for axis in (1, 2, 3):
        if np.random.rand() < 0.5:
            x, others = x.flip(axis), [o.flip(axis) for o in others]
    gain, shift = np.random.uniform(0.8, 1.2), np.random.uniform(-0.1, 0.1)
    return ((x * gain + shift).clamp(0, 1), *others)


# ---- DETNet


def heat_targets(boxes, grid, stride=DETNet.STRIDE):
    """Gaussian centre heatmap (sigma = length / 6, at least one cell) with 1 at the nearest cell, and the cell positions."""
    axes = [(torch.arange(n, device=boxes.device) + 0.5) * stride for n in grid]
    cells = torch.stack(torch.meshgrid(*axes, indexing="ij"))          # (3, *grid)
    sigma = (boxes[:, 3:] / 6).clamp(min=stride).view(-1, 3, 1, 1, 1)
    heat = torch.exp(-0.5 * (((cells[None] - boxes[:, :3].view(-1, 3, 1, 1, 1)) / sigma) ** 2).sum(1))
    peak = heat.flatten(1).argmax(1)
    heat.flatten(1)[torch.arange(len(boxes)), peak] = 1
    return heat, cells


def heat_focal(logit, heat, alpha=2, beta=4):
    """CornerNet's penalty-reduced focal loss, per positive cell."""
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = heat == 1
    loss = -(1 - p) ** alpha * p.log() * pos - (1 - heat) ** beta * p ** alpha * (1 - p).log() * ~pos
    return loss.sum() / pos.sum().clamp(min=1)


def det_loss(out, boxes):
    """Heatmap focal loss, and smooth L1 on the offset (/ 8) and log length where the target heat exceeds 0.5."""
    heat, cells = heat_targets(boxes, out.shape[2:])
    near = (heat > 0.5)[:, None]
    offset = (boxes[:, :3].view(-1, 3, 1, 1, 1) - cells[None]) / 8
    size = boxes[:, 3:].log().view(-1, 3, 1, 1, 1).expand_as(offset)
    reg = F.smooth_l1_loss(out[:, 1:4], offset, reduction="none") + F.smooth_l1_loss(out[:, 4:7], size, reduction="none")
    return heat_focal(out[:, 0], heat) + (reg * near).sum() / (3 * near.sum().clamp(min=1))


def decode(out, k=TOP_K, stride=DETNet.STRIDE):
    """Top local maxima of the heatmap as (n, k, 6) boxes (centre, length) and (n, k) scores."""
    p = torch.sigmoid(out[:, :1])
    peaks = (p * (p == F.max_pool3d(p, 3, 1, 1))).flatten(1)
    scores, index = peaks.topk(k, 1)
    grid = torch.as_tensor(out.shape[2:], device=out.device)
    cell = torch.stack([index // (grid[1] * grid[2]), index // grid[2] % grid[1], index % grid[2]], 2)
    regress = out[:, 1:].flatten(2).transpose(1, 2).gather(1, index[..., None].expand(-1, -1, 6))
    return torch.cat([(cell + 0.5) * stride + regress[..., :3] * 8, regress[..., 3:].clamp(max=8).exp()], 2), scores


def det_batch(volumes, boxes, multiple=8):
    """Whole volumes at 1 mm, padded at the end to one shape: (n, a, b, c), and boxes in that grid."""
    x = [F.avg_pool3d(v[None].float() / 255, 2)[0] for v in volumes]
    shape = [int(np.ceil(max(a.shape[i] for a in x) / multiple) * multiple) for i in range(3)]
    x = torch.stack([F.pad(a, [p for n, s in zip(reversed(a.shape), reversed(shape)) for p in (0, s - n)]) for a in x])
    return x, torch.as_tensor(boxes, device=x.device) / 2


def train_det(data, fit, val, name, out_dir, settings):
    net, x, boxes = DETNet().to(DEVICE), *det_batch([data["vol"][i] for i in fit], data["box"][fit])
    val_x, val_boxes = det_batch([data["vol"][i] for i in val], data["box"][val])

    def step():
        pick = torch.randperm(len(x))[:4]
        v, b = x[pick], boxes[pick].clone()
        for axis in range(3):
            if np.random.rand() < 0.5:
                v, b[:, axis] = v.flip(axis + 1), v.shape[axis + 1] - b[:, axis]
        return det_loss(net(v[:, None]), b)

    def validate():
        found = detect(net, val_x)[0][:, 0].cpu().numpy()
        return float(np.mean([challenge.iou_3d(a, b) for a, b in zip(found, val_boxes.cpu().numpy())]))

    return net if train(net, step, validate, name, out_dir, settings) else None


def detect(net, x, batch=4):
    net.eval()
    with torch.no_grad():
        found = [decode(net(x[i:i + batch, None])) for i in range(0, len(x), batch)]
    return torch.cat([a for a, _ in found]), torch.cat([b for _, b in found])


# ---- SEGNet


def dice_bce(logit, mask):
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * mask).flatten(1).sum(1) + 1) / (p.flatten(1).sum(1) + mask.flatten(1).sum(1) + 1)
    return dice.mean() + F.binary_cross_entropy_with_logits(logit, mask)


def seg_centre(box, shape, patch, foreground):
    """A patch centre: half the time inside the lesion box, otherwise anywhere."""
    if foreground:
        return box[:3] + (np.random.rand(3) - 0.5) * box[3:]
    return np.random.rand(3) * np.asarray(shape)


def windows(shape, patch):
    """Patch corners overlapping by half and covering the shape; a short axis gets one padded window."""
    corner = lambda n, s: [0] if n <= s else sorted({*range(0, n - s, s // 2), n - s})
    return [(a, b, c) for a in corner(shape[0], patch[0]) for b in corner(shape[1], patch[1]) for c in corner(shape[2], patch[2])]


def segment(net, vol, patch=PATCH, batch=8):
    """Sliding-window probability averaged over windows, thresholded, largest component: bool mask of vol's shape."""
    net.eval()
    x = vol.to(DEVICE).float() / 255
    total, count = torch.zeros_like(x), torch.zeros_like(x)
    spots = windows(vol.shape, patch)
    with torch.no_grad():
        for i in range(0, len(spots), batch):
            corners = spots[i:i + batch]
            parts = [crop(x, [c + s / 2 for c, s in zip(corner, patch)], patch) for corner in corners]
            prob = torch.sigmoid(net(torch.stack(parts)[:, None]))
            for corner, p in zip(corners, prob):
                box = tuple(slice(c, min(c + s, n)) for c, s, n in zip(corner, patch, x.shape))
                total[box] += p[tuple(slice(0, b.stop - b.start) for b in box)]
                count[box] += 1
    mask = (total / count > 0.5).cpu().numpy()
    labels, n = ndimage.label(mask)
    return labels == np.bincount(labels.ravel())[1:].argmax() + 1 if n > 1 else mask


def dice(a, b):
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def train_seg(data, fit, val, name, out_dir, settings, patch=PATCH, batch=4):
    net = SEGNet().to(DEVICE)
    vols = {i: data["vol"][i].to(DEVICE) for i in fit}
    masks = {i: data["mask"][i].to(DEVICE) for i in fit}

    def step():
        vs, ms = [], []
        for i in np.random.choice(fit, batch):
            c = seg_centre(data["box"][i], vols[i].shape, patch, np.random.rand() < 0.5)
            vs.append(crop(vols[i], c, patch)), ms.append(crop(masks[i], c, patch))
        v, m = augment(torch.stack(vs).float() / 255, torch.stack(ms).float())
        return dice_bce(net(v[:, None]), m)

    def validate():
        return float(np.mean([dice(segment(net, data["vol"][i], patch), data["mask"][i].numpy()) for i in val]))

    return net if train(net, step, validate, name, out_dir, settings) else None


# ---- CLSNet


def train_cls(data, fit, val, name, out_dir, settings, patch=PATCH, batch=8, jitter=12):
    """Patches around the true lesion centre, moved by up to `jitter` voxels, as a stand-in for detection errors."""
    net = CLSNet().to(DEVICE)
    vols = {i: data["vol"][i].to(DEVICE) for i in fit}

    def step():
        pick = np.random.choice(fit, batch)
        moved = [data["box"][i][:3] + np.random.randint(-jitter, jitter + 1, 3) for i in pick]
        x = augment(torch.stack([crop(vols[i], c, patch) for i, c in zip(pick, moved)]).float() / 255)[0]
        return F.cross_entropy(net(x[:, None]), torch.as_tensor(data["y"][pick], device=x.device))

    def validate():
        return roc_auc_score(data["y"][val], classify(net, data, val, data["box"][val][:, :3], patch))

    return net if train(net, step, validate, name, out_dir, settings) else None


def classify(net, data, cases, centres, patch=PATCH, batch=8):
    """p(malignant) for the patch around centres[k] of volume cases[k]."""
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(cases), batch):
            x = torch.stack([crop(data["vol"][j], c, patch) for j, c in zip(cases[i:i + batch], centres[i:i + batch])])
            out.append(F.softmax(net(x.to(DEVICE).float()[:, None] / 255), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


# ---- shared training and the run


def save_atomic(state, path):
    """A stop during the write leaves the previous checkpoint intact."""
    torch.save(state, f"{path}.tmp")
    os.replace(f"{path}.tmp", path)


def train(net, step, validate, name, out_dir, settings):
    """Keep the best epoch on validation, halve lr on a 5-epoch plateau, stop after `patience` or `max_seconds` of training.

    Resumes from its checkpoint. Returns False if `deadline` (time.time()) came first; otherwise net holds the best weights.
    """
    opt = torch.optim.Adam(net.parameters(), settings["lr"], weight_decay=1e-5)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=5)
    state = dict(epoch=0, best=None, best_score=-1.0, best_epoch=0, seconds=0.0, history=[], done=False)
    checkpoint = Path(out_dir) / f"luo2025_t1_{name}.ckpt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        net.load_state_dict(saved["net"]), opt.load_state_dict(saved["opt"]), plateau.load_state_dict(saved["plateau"])
        np.random.set_state(saved["numpy"]), state.update(saved["state"])
        log(out_dir, f"{name} resumed after epoch {state['epoch']}")
    device = f"cuda:{torch.cuda.current_device()}" if DEVICE == "cuda" else DEVICE
    while not state["done"]:
        if settings.get("deadline") and time.time() > settings["deadline"]:
            log(out_dir, f"{name} stopped at the deadline after epoch {state['epoch']}")
            return False
        started, epoch = time.time(), state["epoch"] + 1
        net.train()
        losses = []
        for _ in range(settings["steps"]):
            loss = step()
            opt.zero_grad(), loss.backward(), opt.step()
            losses.append(loss.item())
        score = float(validate())
        plateau.step(score)
        if state["best"] is None or score > state["best_score"]:     # a NaN score never replaces earlier weights
            state.update(best={k: v.cpu().clone() for k, v in net.state_dict().items()}, best_score=score, best_epoch=epoch)
        state["seconds"] += time.time() - started
        state["epoch"] = epoch
        state["history"].append(dict(epoch=epoch, loss=np.mean(losses), val=score, lr=opt.param_groups[0]["lr"], seconds=state["seconds"]))
        state["done"] = epoch >= settings["epochs"] or epoch - state["best_epoch"] >= settings["patience"] or state["seconds"] > settings["max_seconds"]
        save_atomic(dict(net=net.state_dict(), opt=opt.state_dict(), plateau=plateau.state_dict(), numpy=np.random.get_state(), state=state), checkpoint)
        pd.DataFrame(state["history"]).to_csv(Path(out_dir) / f"luo2025_t1_{name}_history.csv", index=False)
        memory = torch.cuda.max_memory_allocated() / 2 ** 30 if DEVICE == "cuda" else 0
        log(out_dir, f"{name} {device} epoch {epoch} loss {np.mean(losses):.3f} lr {opt.param_groups[0]['lr']:.1e} val {score:.3f} "
                     f"best {state['best_score']:.3f} at {state['best_epoch']} {state['seconds']:.0f} s, {memory:.1f} GiB")
    net.load_state_dict(state["best"])
    torch.save(state["best"], Path(out_dir) / f"luo2025_t1_{name}.pt")
    log(out_dir, f"{name} done, best epoch {state['best_epoch']}, val {state['best_score']:.3f}")
    return True


def load_cache(path=CACHE):
    """y, split, shape, box as arrays; vol and mask as lists of tensors, since volumes differ in shape."""
    d = np.load(path)
    n = len(d["y"])
    return dict(y=d["y"], split=d["split"], source=d["source"], shape=d["shape"], box=d["box"],
                vol=[torch.as_tensor(d[f"vol{i}"]) for i in range(n)], mask=[torch.as_tensor(d[f"mask{i}"]) for i in range(n)])


def run_job(job):
    """One training and its Test predictions, in a GPU worker. A finished job is not run again."""
    task, seed, out_dir = job["task"], job["seed"], job["out_dir"]
    name, patch = f"{task}_seed{seed}", job.get("patch", PATCH)
    out = Path(out_dir) / f"luo2025_t1_{name}.npz"
    if out.exists():
        return
    torch.manual_seed(seed), np.random.seed(seed)
    data = load_cache(job["cache"])
    fit, val, test = [np.flatnonzero(data["split"] == k) for k in range(3)]
    settings = {**TASKS[task], "max_seconds": job["max_seconds"], "deadline": job.get("deadline"), **job.get("overrides", {})}
    log(out_dir, f"{name} start, {len(fit)} fit, {len(val)} validation, {len(test)} test")
    if task == "det":
        net = train_det(data, fit, val, name, out_dir, settings)
        if net is None:
            return
        boxes, scores = detect(net, det_batch([data["vol"][i] for i in test], data["box"][test])[0])
        np.savez_compressed(out, boxes=boxes.cpu().numpy() * 2, scores=scores.cpu().numpy())      # in the cache's 0.5 mm grid
    elif task == "seg":
        net = train_seg(data, fit, val, name, out_dir, settings, patch)
        if net is None:
            return
        np.savez_compressed(out, **{f"mask{k}": np.packbits(segment(net, data["vol"][i], patch)) for k, i in enumerate(test)})
    else:
        net = train_cls(data, fit, val, name, out_dir, settings, patch)
        if net is None:
            return
        centres = np.load(Path(out_dir) / f"luo2025_t1_det_seed{seed}.npz")["boxes"][:, 0, :3]
        np.savez_compressed(out, prob=classify(net, data, test, centres, patch), prob_gt=classify(net, data, test, data["box"][test][:, :3], patch))
    (Path(out_dir) / f"luo2025_t1_{name}.ckpt").unlink(missing_ok=True)
    log(out_dir, f"{name} predictions saved")


def to_original(mask, shape):
    """Nearest-neighbour upsampling of a cache-grid mask to the original voxel grid."""
    return mask[np.ix_(*[np.minimum(((np.arange(n) + 0.5) * m / n).astype(int), m - 1) for n, m in zip(shape, mask.shape)])]


def challenge_box(box, cache_shape, shape):
    """A cache-grid array-axis box as the challenge's (X, Y, Z, x, y, z) in original voxels; X, Y, Z are axes 2, 1, 0."""
    scale = np.asarray(shape) / np.asarray(cache_shape)
    return (*(box[:3] * scale)[::-1], *(box[3:] * scale)[::-1])


def read_test_mask(i):
    return TDSC(path=TDSC_PATH, split=DataSplits.TEST, download=False)[i][1]


def score_case(load_mask, k, source, cache_shape, packed):
    """Ground-truth box, {seed: (dice, hd)} and original shape of one test volume."""
    mask = np.asarray(load_mask(source)) > 0
    pred = lambda bits: np.unpackbits(bits, count=int(np.prod(cache_shape))).reshape(cache_shape).astype(bool)
    return challenge.mask_box(mask), {s: challenge.dice_hd(mask, to_original(pred(b), mask.shape)) for s, b in packed.items()}, mask.shape


def evaluate(data, seeds, out_dir, load_mask=read_test_mask):
    """The protocol's and the challenge's metrics, per seed and over seeds. Dice and HD are on the original grid, in voxels."""
    test = np.flatnonzero(data["split"] == 2)
    y, rows, out_dir = data["y"][test], {}, Path(out_dir)
    runs = {s: dict(det=np.load(out_dir / f"luo2025_t1_det_seed{s}.npz"), seg=np.load(out_dir / f"luo2025_t1_seg_seed{s}.npz"),
                    cls=np.load(out_dir / f"luo2025_t1_cls_seed{s}.npz")) for s in seeds}
    work = [(k, int(data["source"][i]), data["vol"][i].shape, {s: runs[s]["seg"][f"mask{k}"] for s in seeds}) for k, i in enumerate(test)]
    done = Parallel(n_jobs=os.cpu_count(), max_nbytes=None)(delayed(score_case)(load_mask, *w) for w in work)   # distance transforms are the slow part
    log(out_dir, "segmentation scored")
    cases = {s: [d[1][s] for d in done] for s in seeds}
    gt = {k: [d[0]] for k, d in enumerate(done)}
    boxes = {s: {k: [(challenge_box(b, work[k][2], done[k][2]), float(p)) for b, p in zip(runs[s]["det"]["boxes"][k], runs[s]["det"]["scores"][k])]
                 for k in gt} for s in seeds}
    for s in seeds:
        np.save(out_dir / f"luo2025_t1_seg_cases_seed{s}.npy", np.array(cases[s]))
        p = runs[s]["cls"]["prob"]
        rows[f"seed {s}"] = {**metrics(y, p), "AUC_gt_centre": roc_auc_score(y, runs[s]["cls"]["prob_gt"]), **challenge.classification(y, p),
                             **challenge.segmentation(cases[s]), "FROC": challenge.froc(gt, boxes[s])["FROC"]}
    table = pd.DataFrame(rows).T
    table.loc["mean of seeds"] = table.mean()
    ensemble = np.mean([runs[s]["cls"]["prob"] for s in seeds], 0)
    ensemble_gt = np.mean([runs[s]["cls"]["prob_gt"] for s in seeds], 0)
    table.loc["ensemble, classification only"] = {**metrics(y, ensemble), "AUC_gt_centre": roc_auc_score(y, ensemble_gt), **challenge.classification(y, ensemble)}
    log(out_dir, f"ensemble AUC 95% CI {auc_ci(y, ensemble)}")
    table = table.round(4)
    table.to_csv(out_dir / "luo2025_t1_results.csv")
    board = challenge.benchmark({"T1 reproduced, mean of seeds": table.loc["mean of seeds"].to_dict()})
    board.to_csv(out_dir / "luo2025_t1_benchmark.csv")
    print(table.to_string()), print(board.T.to_string())
    return table


def main(seeds=(0, 1, 2), out_dir=".", max_seconds=5400, hours=None):
    """`hours` limits training, so that a kernel ends cleanly and keeps its output before the platform stops it."""
    deadline = None if hours is None else time.time() + hours * 3600
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if not CACHE.exists():
        log(out_dir, "building the cache")
        build_cache()
        log(out_dir, "cache built")
    log(out_dir, f"{torch.cuda.device_count()} GPUs, {os.cpu_count()} CPUs")
    job = lambda task, s: dict(task=task, seed=s, out_dir=str(out_dir), cache=str(CACHE), max_seconds=max_seconds, deadline=deadline)
    finished = lambda task: all((Path(out_dir) / f"luo2025_t1_{task}_seed{s}.npz").exists() for s in seeds)
    run_on_gpus(run_job, [job(t, s) for s in seeds for t in ("det", "seg")])
    if not (finished("det") and finished("seg")):
        return log(out_dir, "detection or segmentation not finished; run again to resume")
    run_on_gpus(run_job, [job("cls", s) for s in seeds])              # patches come from the same seed's detections
    if not finished("cls"):
        return log(out_dir, "classification not finished; run again to resume")
    return evaluate(load_cache(), seeds, out_dir)
