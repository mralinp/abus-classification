"""Reproduce Barekatrezaei et al., ESWA 2025: Laplace-Beltrami spectra + dual-path CNN + stacking.

    uv run python reproduce_eswa2025.py            # features -> 5-fold CV -> test table
    uv run python reproduce_eswa2025.py --epochs 5 # quick smoke run

Positive class = malignant (TDSC label 0). Table 3 of the paper is printed at the end.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy import ndimage
from scipy.sparse.linalg import eigsh
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.manifold import Isomap
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tdsc_abus2023_pytorch import TDSC, DataSplits
from torch import nn

from abus_classification.features.radiology import extract_radiology_features
from abus_classification.features.shape_descriptor.shape_descriptors import compute_laplace_beltrami_operator
from abus_classification.utils.mesh import lesion_mesh
from abus_classification.utils.spacing import TDSC_SPACING

ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "eswa2025.npz"
TDSC_PATH = os.environ.get("TDSC_PATH", str(ROOT / "data" / "tdsc"))  # downloaded here if missing
PATCH, ISO, N_EIG = (128, 64, 80), 0.476, 100
# Tan et al. 2012 six features, computed with the repo's millimetre-aware extractor.
RADIOMIC = ["height_to_width_ratio", "entropy", "mean_intensity", "margin_contrast", "intensity_variance", "sphericity"]
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def lb_eigenvalues(mask):
    """First 100 eigenvalues of the cotangent Laplacian L (Eq. 3-4, no mass matrix), mesh in mm."""
    pts = np.argwhere(mask)
    lo, hi = pts.min(0), pts.max(0) + 1
    mesh = lesion_mesh(mask[tuple(slice(a, b) for a, b in zip(lo, hi))], TDSC_SPACING)
    L, _ = compute_laplace_beltrami_operator(mesh)
    return np.sort(eigsh(L, k=N_EIG, sigma=-1e-8, which="LM", return_eigenvectors=False))


def patch(vol, mask):
    """128x64x80 VOI and mask at 0.476 mm isotropic voxels, centred on the lesion."""
    centre = np.argwhere(mask).mean(0)
    scale = ISO / np.asarray(TDSC_SPACING)                 # source voxels per output voxel
    half = np.asarray(PATCH) * scale / 2
    lo = np.maximum(np.floor(centre - half).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(centre + half).astype(int) + 3, vol.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    offset = centre - half + 0.5 * scale - 0.5 - lo
    out = []
    for a in (vol[box], mask[box]):                        # anti-alias, then spline-resample
        a = ndimage.gaussian_filter(a.astype(np.float32), 0.5 * scale)
        out.append(ndimage.affine_transform(a, scale, offset=offset, output_shape=PATCH, order=1))
    return (out[0] / 255).astype(np.float16), (out[1] > 0.5).astype(np.float16)


def one_case(split, i):
    vol, mask, label, _ = TDSC(path=TDSC_PATH, split=split, download=False)[i]
    v, m = patch(vol, mask)
    feats = extract_radiology_features(vol, mask)           # once: it computes every feature
    return dict(split=split.name == "TEST", y=int(label == 0),   # y: 1 = malignant
                radiomic=[feats[k] for k in RADIOMIC], lb=lb_eigenvalues(mask), vol=v, mask=m)


def build_cache():
    CACHE.parent.mkdir(parents=True, exist_ok=True)         # fail now, not after hours of work
    jobs = []
    for split in (DataSplits.TRAIN, DataSplits.VALIDATION, DataSplits.TEST):
        ds = TDSC(path=TDSC_PATH, split=split, download=not (Path(TDSC_PATH) / str(split)).exists())
        jobs += [(split, i) for i in range(len(ds))]
    cases = Parallel(n_jobs=-1, verbose=5)(delayed(one_case)(*j) for j in jobs)   # input order is kept
    np.savez_compressed(CACHE, **{k: np.asarray([c[k] for c in cases]) for k in cases[0]})


class DualPath(nn.Module):
    """Two conv paths (VOI, mask) -> concat -> GAP -> + LB eigenvalues -> dense -> sigmoid."""

    def __init__(self, n_lb, widths=(16, 32, 64, 128)):
        super().__init__()

        def path():
            layers, c = [], 1
            for w in widths:
                for _ in range(2):
                    layers += [nn.Conv3d(c, w, 3, padding=1), nn.InstanceNorm3d(w), nn.ReLU()]
                    c = w
                layers.append(nn.MaxPool3d(2))
            return nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1), nn.Flatten())

        self.vol, self.mask = path(), path()
        self.head = nn.Sequential(nn.Linear(2 * widths[-1] + n_lb, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, vol, mask, lb):
        return self.head(torch.cat([self.vol(vol), self.mask(mask), lb], 1)).squeeze(1)


def fit_cnn(tr, te_sets, epochs, seed):
    """Train on indices `tr` of (vol, mask, lb, y); return sigmoid outputs for each set in `te_sets`."""
    torch.manual_seed(seed)
    net = DualPath(tr[2].shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32)).to(DEVICE)
    vol, mask, lb, y = map(t, tr)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(y)).split(8):
            v, m = vol[idx][:, None], mask[idx][:, None]
            for ax in (2, 4):                                # random flips (axis 0 and 2 of the patch)
                if np.random.rand() < 0.5:
                    v, m = v.flip(ax), m.flip(ax)
            loss = nn.functional.binary_cross_entropy_with_logits(net(v, m, lb[idx]), y[idx])
            opt.zero_grad(), loss.backward(), opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        return [torch.sigmoid(net(t(v)[:, None], t(m)[:, None], t(l))).cpu().numpy() for v, m, l in te_sets]


def metrics(y, p):
    pred = p >= 0.5
    return dict(AUC=roc_auc_score(y, p), Acc=accuracy_score(y, pred), Prec=precision_score(y, pred),
                Sens=recall_score(y, pred), Spec=recall_score(y, pred, pos_label=0), F1=f1_score(y, pred))


def main(epochs, seed=0):
    if not CACHE.exists():
        build_cache()
    d = np.load(CACHE)
    test, y = d["split"], d["y"]
    X = {k: d[k] for k in ("radiomic", "lb", "vol", "mask")}
    tv, te = np.flatnonzero(~test), np.flatnonzero(test)
    # Eigenvalues span orders of magnitude; standardise their logs before Isomap / the CNN head.
    X["lb"] = np.log10(np.maximum(X["lb"], 1e-12)[:, 1:])   # drop lambda_0 = 0
    lb_std = StandardScaler().fit(X["lb"][tv])
    X["lb"] = lb_std.transform(X["lb"])

    names = ("SVM", "ExtraTrees", "CNN")
    oof, tst = np.zeros((3, len(tv))), np.zeros((3, len(te)))
    for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(tv, y[tv])):
        a, b = tv[a], tv[b]
        iso = Isomap(n_components=2, n_neighbors=5).fit(X["lb"][a])
        tab = lambda i: np.c_[X["radiomic"][i], iso.transform(X["lb"][i])]
        for k, clf in enumerate((SVC(kernel="linear", probability=True, random_state=seed),
                                 ExtraTreesClassifier(500, random_state=seed))):
            m = make_pipeline(StandardScaler(), clf).fit(tab(a), y[a])
            oof[k, np.isin(tv, b)], tst[k] = m.predict_proba(tab(b))[:, 1], tst[k] + m.predict_proba(tab(te))[:, 1] / 5
        pack = lambda i: (X["vol"][i], X["mask"][i], X["lb"][i])
        p_b, p_te = fit_cnn((*pack(a), y[a]), [pack(b), pack(te)], epochs, seed + fold)
        oof[2, np.isin(tv, b)], tst[2] = p_b, tst[2] + p_te / 5
        print(f"fold {fold} done", flush=True)

    meta = HistGradientBoostingClassifier(random_state=seed).fit(oof.T, y[tv])
    rows = {n: metrics(y[te], tst[i]) for i, n in enumerate(names)}
    rows["Voting"] = metrics(y[te], tst.mean(0))
    rows["Stacking"] = metrics(y[te], meta.predict_proba(tst.T)[:, 1])
    table = pd.DataFrame(rows).T.round(4)
    print(table.to_string())
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    main(**vars(ap.parse_args()))
