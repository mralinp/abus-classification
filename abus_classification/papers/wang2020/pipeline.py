"""Wang et al., Ultrasound Med. Biol. 46(5):1119-1132 (2020): multiview Inception-v3 on ABUS lesion patches.

    python -m abus_classification.papers.wang2020       # builds data/wang2020.npz from TDSC_PATH; train on a GPU

Paper details followed: patches cropped along the lesion's bounding box in coronal and transverse views; multiview CNN A
stacks coronal, transverse, transverse as image channels, for every combination of slices; CNN B has one backbone per
view; ImageNet Inception-v3, all layers fine-tuned; FC 256-128-2 with dropout after the first two, Xavier uniform;
Adadelta, batch 32; rotations and flips; early stopping when validation AUC has not increased for 5 epochs.
Deviations: official TDSC bounding boxes (paper: physicians' boxes); two slices per view at 1/3 and 2/3 of the box, which
gives 8 CNN A patches per lesion (paper: chosen slices, about 10 per lesion); Train (100) to fit and Validation (30) for
early stopping, reported on the official Test split (paper: five-fold CV); random rotation and flip per sample (paper:
fixed copies); not given and assumed: resize to 299 x 299 ignoring aspect ratio, cross-entropy, dropout 0.5, Adadelta lr 1.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from tdsc_abus2023_pytorch import TDSC, DataSplits

from abus_classification.data.lesion_cache import ROOT, TDSC_PATH
from abus_classification.papers.wang2020.model import WangCNN
from abus_classification.protocol import DEVICE, auc_ci, metrics

CACHE, SIZE = ROOT / "data" / "wang2020.npz", 299
# Slices per view: indices 0, 1 are coronal and 2, 3 transverse. A combination gives the channels of one input patch.
MODES = {
    "cnn_a": [(c, t, u) for c in (0, 1) for t in (2, 3) for u in (2, 3)],
    "cnn_b": [(c, c, c, t, t, t) for c in (0, 1) for t in (2, 3)],
    "coronal": [(0, 0, 0), (1, 1, 1)],
    "transverse": [(2, 2, 2), (3, 3, 3)],
    "cnn_a_scratch": [(c, t, u) for c in (0, 1) for t in (2, 3) for u in (2, 3)],
}


def lesion_patches(vol, box, size=SIZE):
    """Coronal and transverse patches, (4, size, size) uint8. box: ((x0, y0, z0), (x1, y1, z1)), x, y, z = axes 2, 1, 0."""
    (x0, y0, z0), (x1, y1, z1) = [[int(v) for v in corner] for corner in box]
    third = lambda a, b: (a + (b - a) // 3, a + 2 * (b - a) // 3)
    crops = [vol[z0:z1 + 1, y, x0:x1 + 1] for y in third(y0, y1)]     # coronal: parallel to the skin, axis 1 is depth
    crops += [vol[z0:z1 + 1, y0:y1 + 1, x] for x in third(x0, x1)]    # transverse: the acquired slices, axis 2
    resize = lambda a: F.interpolate(torch.as_tensor(np.asarray(a, np.float32))[None, None], (size, size), mode="bilinear",
                                     align_corners=False, antialias=True)[0, 0].clamp(0, 255).round().numpy().astype(np.uint8)
    return np.stack([resize(c) for c in crops])


def build_cache():
    rows = []
    for code, split in enumerate((DataSplits.TRAIN, DataSplits.VALIDATION, DataSplits.TEST)):
        ds = TDSC(path=TDSC_PATH, split=split, download=False)
        for i in range(len(ds)):
            vol, _, label, box = ds[i]
            rows.append((lesion_patches(vol, box), int(label == 0), code))    # label 0 = malignant
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, patches=np.stack([r[0] for r in rows]), y=np.array([r[1] for r in rows]),
                        split=np.array([r[2] for r in rows]))


def rotate_flip(x):
    """A random rotation by a multiple of 90 degrees and random flips, drawn per sample, the same for all channels."""
    return torch.stack([torch.rot90(s, int(np.random.randint(4)), (1, 2)).flip([d for d in (1, 2) if np.random.rand() < 0.5])
                        for s in x])


def predict(net, patches, combos, batch=32):
    """p(malignant) per lesion: the mean over its combinations."""
    net.eval()
    x = [patches[:, list(c)] for c in combos]
    out = np.zeros(len(patches))
    with torch.no_grad():
        for v in x:
            for idx in np.array_split(np.arange(len(v)), max(1, len(v) // batch)):
                logits = net(torch.as_tensor(v[idx], dtype=torch.float32, device=DEVICE) / 255)
                out[idx] += F.softmax(logits, 1)[:, 1].cpu().numpy() / len(x)
    return out


def train_run(data, mode, seed, out_dir=".", max_epochs=50, patience=5, batch=32):
    """Train one configuration and save its results. A finished run is loaded, never trained again."""
    out = Path(out_dir) / f"wang2020_{mode}_seed{seed}.npz"
    if out.exists():
        return np.load(out)["prob"]
    torch.manual_seed(seed), np.random.seed(seed)
    patches, y, split, combos = data["patches"], data["y"], data["split"], MODES[mode]
    tr, va, te = [np.flatnonzero(split == k) for k in range(3)]
    net = WangCNN(len(combos[0]), pretrained=mode != "cnn_a_scratch").to(DEVICE)
    opt = torch.optim.Adadelta(net.parameters())
    pairs = [(i, c) for i in tr for c in combos]
    best, best_auc, since, history = None, -1.0, 0, []
    for epoch in range(1, max_epochs + 1):
        net.train()
        order, losses = np.random.permutation(len(pairs)), []
        for start in range(0, len(order), batch):
            chosen = [pairs[k] for k in order[start:start + batch]]
            if len(chosen) < 2:                                       # BatchNorm needs more than one sample
                continue
            x = torch.as_tensor(np.stack([patches[i, list(c)] for i, c in chosen]), dtype=torch.float32) / 255
            target = torch.as_tensor(y[[i for i, _ in chosen]], device=DEVICE)
            loss = F.cross_entropy(net(rotate_flip(x).to(DEVICE)), target)
            opt.zero_grad(), loss.backward(), opt.step()
            losses.append(loss.item())
        auc = roc_auc_score(y[va], predict(net, patches[va], combos, batch))
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), val_auc=auc))
        print(f"{mode} seed {seed} epoch {epoch} loss {history[-1]['loss']:.3f} val AUC {auc:.3f}", flush=True)
        if auc > best_auc:
            best_auc, since = auc, 0
            best = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    net.load_state_dict(best)
    prob = predict(net, patches[te], combos, batch)
    pd.DataFrame(history).to_csv(Path(out_dir) / f"wang2020_{mode}_seed{seed}_history.csv", index=False)
    torch.save(net.state_dict(), Path(out_dir) / f"wang2020_{mode}_seed{seed}.pt")
    np.savez_compressed(out, prob=prob, val_auc=best_auc)
    return prob


def main(modes=tuple(MODES), seeds=(0, 1, 2), out_dir=".", max_epochs=50):
    if not CACHE.exists():
        build_cache()
    data = dict(np.load(CACHE))
    y = data["y"][data["split"] == 2]
    rows = {}
    for mode in modes:
        probs = [train_run(data, mode, s, out_dir, max_epochs) for s in seeds]
        rows.update({f"{mode} seed {s}": metrics(y, p) for s, p in zip(seeds, probs)})
        mean = np.mean(probs, 0)
        rows[f"{mode} mean of seeds"] = metrics(y, mean)
        print(mode, "ensemble AUC 95%% CI %.3f-%.3f" % tuple(auc_ci(y, mean)), flush=True)
        table = pd.DataFrame(rows).T.round(4)
        table.to_csv(Path(out_dir) / "wang2020_results.csv")          # keep finished configurations
    print(table.to_string())
    return table
