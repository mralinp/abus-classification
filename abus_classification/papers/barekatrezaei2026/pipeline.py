"""Barekatrezaei et al. (ESWA 2026): SVM + Extra Trees + dual-path CNN, combined by voting or stacking.

Features: 6 radiomic (Tan 2012) + 100 Laplace-Beltrami eigenvalues (Isomap to 2-D for the tabular models, full for the
CNN). 5-fold CV on Train+Validation gives out-of-fold predictions; test predictions average the 5 fold models.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from abus_classification.data.lesion_cache import CACHE, build_cache, load_raw
from abus_classification.papers.barekatrezaei2026.model import fit_cnn
from abus_classification.protocol import folds, metrics


def load():
    """Cached features with LB eigenvalues log-standardised (they span orders of magnitude; lambda_0 = 0 dropped)."""
    d, tv, te = load_raw()
    lb = np.log10(np.maximum(d["lb"], 1e-12)[:, 1:])
    X = dict(radiomic=d["radiomic"], lb=StandardScaler().fit(lb[tv]).transform(lb), vol=d["vol"], mask=d["mask"])
    return X, d["y"], tv, te


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
