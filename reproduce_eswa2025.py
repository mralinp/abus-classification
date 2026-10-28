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
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
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


def load():
    """Cached features with LB eigenvalues log-standardised (they span orders of magnitude; lambda_0 = 0 dropped)."""
    if not CACHE.exists():
        build_cache()
    d = np.load(CACHE)
    test, y = d["split"], d["y"]
    tv, te = np.flatnonzero(~test), np.flatnonzero(test)
    lb = np.log10(np.maximum(d["lb"], 1e-12)[:, 1:])
    X = dict(radiomic=d["radiomic"], lb=StandardScaler().fit(lb[tv]).transform(lb), vol=d["vol"], mask=d["mask"])
    return X, y, tv, te


def folds(y, tv, seed):
    return [(tv[a], tv[b]) for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tv, y[tv])]


def tabular_cv(X, y, tv, te, seed, rad=True, lb=True, reducer="isomap", dim=2):
    """Out-of-fold / fold-averaged test probabilities of SVM and Extra Trees on radiomic + reduced LB features."""
    red = dict(isomap=lambda: Isomap(n_components=dim, n_neighbors=5), pca=lambda: PCA(dim, random_state=seed),
               kpca=lambda: KernelPCA(dim, kernel="rbf", random_state=seed),
               lle=lambda: LocallyLinearEmbedding(n_components=dim, n_neighbors=5, random_state=seed),
               none=lambda: FunctionTransformer())[reducer]
    oof, tst = np.zeros((2, len(tv))), np.zeros((2, len(te)))
    for a, b in folds(y, tv, seed):
        r = red().fit(X["lb"][a]) if lb else None
        empty = lambda i: np.empty((len(i), 0))
        tab = lambda i: np.c_[X["radiomic"][i] if rad else empty(i), r.transform(X["lb"][i]) if lb else empty(i)]
        for k, clf in enumerate((SVC(kernel="linear", probability=True, random_state=seed), ExtraTreesClassifier(500, random_state=seed))):
            m = make_pipeline(StandardScaler(), clf).fit(tab(a), y[a])
            oof[k, np.isin(tv, b)], tst[k] = m.predict_proba(tab(b))[:, 1], tst[k] + m.predict_proba(tab(te))[:, 1] / 5
    return oof, tst


def cnn_cv(X, y, tv, te, epochs, seed, head="lb"):
    """Dual-path CNN; `head` picks the features concatenated after GAP: lb, both (radiomic + lb) or none."""
    ext = dict(lb=X["lb"], both=np.c_[X["lb"], StandardScaler().fit(X["radiomic"][tv]).transform(X["radiomic"])],
               none=np.empty((len(y), 0)))[head]
    pack = lambda i: (X["vol"][i], X["mask"][i], ext[i])
    oof, tst = np.zeros(len(tv)), np.zeros(len(te))
    for fold, (a, b) in enumerate(folds(y, tv, seed)):
        p_b, p_te = fit_cnn((*pack(a), y[a]), [pack(b), pack(te)], epochs, seed + fold)
        oof[np.isin(tv, b)], tst = p_b, tst + p_te / 5
        print(f"cnn[{head}] fold {fold} done", flush=True)
    return oof, tst


def stack(oof, tst, y_tv, y_te):
    """Voting plus two stackers (gradient boosting, logistic regression) over the base classifiers' outputs."""
    rows = {"Voting": metrics(y_te, tst.mean(0))}
    for n, meta in (("Stacking-HGB", HistGradientBoostingClassifier(random_state=0)), ("Stacking-LR", LogisticRegression())):
        rows[n] = metrics(y_te, meta.fit(oof.T, y_tv).predict_proba(tst.T)[:, 1])
    return rows


def main(epochs, seed=0):
    X, y, tv, te = load()
    (o_t, t_t), (o_c, t_c) = tabular_cv(X, y, tv, te, seed), cnn_cv(X, y, tv, te, epochs, seed)
    oof, tst = np.r_[o_t, o_c[None]], np.r_[t_t, t_c[None]]
    np.savez(CACHE.with_name(f"eswa2025_preds_s{seed}.npz"), oof=oof, tst=tst, y_tv=y[tv], y_te=y[te])
    rows = {n: metrics(y[te], tst[i]) for i, n in enumerate(("SVM", "ExtraTrees", "CNN"))}
    rows.update(stack(oof, tst, y[tv], y[te]))
    table = pd.DataFrame(rows).T.round(4)
    print(table.to_string())
    return table


def ablations(epochs, seed=0):
    """Tables 4-8 of the paper; the CNN is only retrained where a config changes it (Table 8)."""
    X, y, tv, te = load()
    cnn = {h: cnn_cv(X, y, tv, te, epochs, seed, h) for h in ("lb", "both", "none")}
    full = tabular_cv(X, y, tv, te, seed)
    run = lambda tab, c: stack(np.r_[tab[0], c[0][None]], np.r_[tab[1], c[1][None]], y[tv], y[te])
    configs = {
        **{f"T4 {r}": run(tabular_cv(X, y, tv, te, seed, reducer=r), cnn["lb"]) for r in ("isomap", "pca", "kpca", "lle", "none")},
        **{f"T5 dim={d}": run(tabular_cv(X, y, tv, te, seed, dim=d), cnn["lb"]) for d in (2, 5, 10, 20, 50)},
        "T7 no LB": run(tabular_cv(X, y, tv, te, seed, lb=False), cnn["lb"]),
        "T7 no radiomic": run(tabular_cv(X, y, tv, te, seed, rad=False), cnn["lb"]),
        "T8 CNN rad+LB": run(full, cnn["both"]),
        "T8 CNN none": run(full, cnn["none"]),
    }
    oof, tst = np.r_[full[0], cnn["lb"][0][None]], np.r_[full[1], cnn["lb"][1][None]]
    for drop, name in enumerate(("SVM", "ExtraTrees", "CNN")):       # T6: leave one base classifier out
        keep = [k for k in range(3) if k != drop]
        configs[f"T6 without {name}"] = stack(oof[keep], tst[keep], y[tv], y[te])
    table = pd.DataFrame({f"{c} [{m}]": v for c, r in configs.items() for m, v in r.items()}).T.round(4)
    print(table.to_string())
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", action="store_true", help="paper Tables 4-8 instead of Table 3")
    a = ap.parse_args()
    (ablations if a.ablate else main)(a.epochs, a.seed)
