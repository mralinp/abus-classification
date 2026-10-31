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
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from sklearn.metrics import cohen_kappa_score
from tdsc_abus2023_pytorch import TDSC, DataSplits
from torch.nn import functional as F

from abus_classification.data.lesion_cache import ROOT, TDSC_PATH
from abus_classification.papers.hejduk2022.model import HEIGHT, WIDTH, HejdukCNN
from abus_classification.protocol import DEVICE, auc_ci, metrics

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


def build_cache():
    """Test volumes only give lesion crops: their windows are evaluated in place."""
    rng, rows = np.random.default_rng(0), {k: [] for k in ("crops", "kind", "y", "split", "case")}
    case = 0
    for code, split in enumerate((DataSplits.TRAIN, DataSplits.VALIDATION, DataSplits.TEST)):
        ds = TDSC(path=TDSC_PATH, split=split, download=False)
        for i in range(len(ds)):
            vol, mask, label, _ = ds[i]
            crops, kinds = volume_crops(vol, mask, rng)
            if code == 2:
                crops, kinds = crops[kinds == LESION], kinds[kinds == LESION]
            for key, v in zip(rows, (crops, kinds, np.full(len(kinds), int(label == 0)), np.full(len(kinds), code), np.full(len(kinds), case))):
                rows[key].append(v)                                   # label 0 = malignant
            case += 1
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **{k: np.concatenate(v) for k, v in rows.items()})


def training_set(data, model):
    """Indices of the crops a model trains on, and their targets, from Train + Validation."""
    fit, kind = data["split"] != 2, data["kind"]
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


def train_model(data, model, seed, out_dir=".", lr=1e-6, batch=16, epochs=None, make=HejdukCNN):
    """Train one of the three models. A finished model is loaded, never trained again."""
    tag = "" if lr == 1e-6 else f"_lr{lr:g}"
    path = Path(out_dir) / f"hejduk2022_model{model}{tag}_seed{seed}.pt"
    net = make().to(DEVICE)
    if path.exists():
        net.load_state_dict(torch.load(path, map_location=DEVICE))
        return net
    torch.manual_seed(seed), np.random.seed(seed)
    idx, y = training_set(data, model)
    crops, target = torch.as_tensor(data["crops"][idx], device=DEVICE), torch.as_tensor(y, device=DEVICE)
    weight = torch.as_tensor(class_weights(y), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True)
    images, epochs, history = MODELS[model]["images"], epochs or MODELS[model]["epochs"], []
    for epoch in range(1, epochs + 1):
        net.train()
        order, losses, hits = np.random.randint(0, len(idx), images), [], 0
        for start in range(0, images, batch):
            pick = order[start:start + batch]
            logits = net(augment(crops[pick, None].float() / 255))
            loss = (F.cross_entropy(logits, target[pick], reduction="none") * weight[target[pick]]).mean()
            opt.zero_grad(), loss.backward(), opt.step()
            losses.append(loss.item())
            hits += (logits.argmax(1) == target[pick]).sum().item()
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), acc=hits / images))
        print(f"model {model} seed {seed} epoch {epoch} loss {history[-1]['loss']:.3f} acc {history[-1]['acc']:.3f}", flush=True)
    pd.DataFrame(history).to_csv(path.with_name(path.stem + "_history.csv"), index=False)
    torch.save(net.state_dict(), path)
    return net


def evaluate(data, nets, seed, out_dir=".", lr=1e-6, stride=10, slice_step=1):
    """Test results of one seed: per lesion crop, then per volume. A finished seed is loaded."""
    tag = "" if lr == 1e-6 else f"_lr{lr:g}"
    out = Path(out_dir) / f"hejduk2022{tag}_seed{seed}_test.npz"
    if out.exists():
        return dict(np.load(out))
    te = np.flatnonzero((data["split"] == 2) & (data["kind"] == LESION))
    image = predict(nets[2], torch.as_tensor(data["crops"][te])).numpy()
    ds = TDSC(path=TDSC_PATH, split=DataSplits.TEST, download=False)
    volume, y, maps, boxes = [], [], {}, []
    for i in range(len(ds)):
        vol, _, label, _ = ds[i]
        m = volume_maps(nets, vol, stride, slice_step)
        volume.append(m[:, 2].max())
        y.append(int(label == 0))
        maps[f"map_{i}"] = (m * 255).astype(np.uint8)
        boxes += [(i, *box, score) for box, score in detect(m, np.shape(vol), stride, slice_step)]
        print(f"seed {seed} volume {i} score {volume[-1]:.3f} found {int((m[:, 1] >= 0.5).sum())} windows", flush=True)
    result = dict(image=image, image_y=data["y"][te], image_case=data["case"][te], volume=np.array(volume), y=np.array(y),
                  boxes=np.array(boxes, float).reshape(-1, 8), stride=stride, slice_step=slice_step, **maps)
    np.savez_compressed(out, **result)
    return result


def row(y, p):
    return {**metrics(y, p), "Kappa": cohen_kappa_score(y, p >= 0.5)}


def main(seeds=(0, 1, 2), out_dir=".", lr=1e-6, stride=10, slice_step=1):
    if not CACHE.exists():
        build_cache()
    data = dict(np.load(CACHE))
    rows, runs, tag = {}, [], "" if lr == 1e-6 else f"_lr{lr:g}"
    for seed in seeds:
        nets = [train_model(data, m, seed, out_dir, lr) for m in MODELS]
        r = evaluate(data, nets, seed, out_dir, lr, stride, slice_step)
        runs.append(r)
        rows[f"images seed {seed}"] = row(r["image_y"], r["image"])
        rows[f"volumes seed {seed}"] = row(r["y"], r["volume"])
        table = pd.DataFrame(rows).T.round(4)
        table.to_csv(Path(out_dir) / f"hejduk2022{tag}_results.csv")      # keep finished seeds
    mean = np.mean([r["volume"] for r in runs], 0)
    rows["volumes mean of seeds"] = row(runs[0]["y"], mean)
    print("volumes ensemble AUC 95%% CI %.3f-%.3f" % tuple(auc_ci(runs[0]["y"], mean)), flush=True)
    table = pd.DataFrame(rows).T.round(4)
    table.to_csv(Path(out_dir) / f"hejduk2022{tag}_results.csv")
    print(table.to_string())
    return table
