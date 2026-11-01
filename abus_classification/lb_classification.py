"""Which shape and image features classify TDSC-ABUS lesions, alone and combined, on Train+Validation only.

Combines the stored Laplace-Beltrami results with Tan's classical features. Every selection runs inside cross-validation,
so its score is not inflated by having picked the winner. The Test split stays untouched: `tv_table` refuses to return it.

    python -m abus_classification.lb_classification
"""
import itertools
import json
import os
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"   # one process per task

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "lb-classification"
TAU_NAMES = ["0.004", "0.007", "0.012", "0.021", "0.037", "0.065", "0.114", "0.200"]
SPECTRAL = ("Shape-DNA", "Heat trace", "HKS local", "WKS local", "GPS", "Tips", "Harmonics", "Hersch ratio")
CLASSICAL = ("Size", "Spiculation", "Intensity", "Classical shape", "Elongation and sphericity")


def build_table():
    """One row per lesion, all 200, in dataset order, with `split` and `y`. Columns are named '<group>: <feature>'."""
    d, s, h = (np.load(RESULTS / p) for p in ("lb-descriptors/features.npz", "lb-stability/spectra.npz", "lb-spharm/features.npz"))
    tan = pd.read_csv(ROOT / "data" / "tan_features.csv")
    assert ((tan.label == 0).astype(int).values == d["y"]).all() and (d["y"] == s["y"]).all() and (d["y"] == h["y"]).all()
    volume, area = d["volume"][:, 0], d["area"][:, 0]
    extents = np.sort(s["extents"][:, 0], -1)
    out = {"split": tan.split.values, "case": tan.case.values, "y": d["y"]}
    def add(group, name, values):
        out[f"{group}: {name}"] = values

    add("Size", "log volume", np.log(volume)), add("Size", "log area", np.log(area))
    for c in ("SPmean", "SP90", "SP75", "SPcyl_mean", "SPcyl_upper_mean", "SPcyl_max_slice_mean"):
        add("Spiculation", c, tan[c].values)
    for c in ("VI", "H", "AI", "IBV", "MC", "PAB"):
        add("Intensity", c, tan[c].values)
    for c in ("VHWR", "SH", "CP", "DF", "DICE"):
        add("Classical shape", c, tan[c].values)
    add("Elongation and sphericity", "log elongation", np.log(extents[:, 2] / extents[:, 0]))
    add("Elongation and sphericity", "log sphericity", np.log(36 * np.pi * volume ** 2 / area ** 3))
    add("Hersch ratio", "lambda_1 * area / 8 pi", s["lam"][:, 0, 1] * s["area"][:, 0] / (8 * np.pi))
    for k in range(20):
        add("Shape-DNA", f"log s_{k + 1}", np.log(d["dna"][:, 0, k]))
    for i, tau in enumerate(TAU_NAMES):
        add("Heat trace", f"log trace, tau {tau}", np.log(d["hks"][:, 0, 0, i]))
        add("HKS local", f"log spread, tau {tau}", np.log(d["hks"][:, 0, 1, i] / d["hks"][:, 0, 0, i]))
        add("HKS local", f"log p95, tau {tau}", np.log(d["hks"][:, 0, 2, i] / d["hks"][:, 0, 0, i]))
        add("WKS local", f"log spread, band {i + 1}", np.log(d["wks"][:, 0, 1, i]))
        add("WKS local", f"log p95, band {i + 1}", np.log(d["wks"][:, 0, 2, i]))
        add("Tips", f"log tips, tau {tau}", np.log(d["tips"][:, 0, i] + 1.0))
    for i, q in enumerate((10, 25, 50, 75, 90, 95, 99)):
        add("GPS", f"log distance p{q}", np.log(d["gps"][:, 0, i]))
    for l in range(1, 16):
        add("Harmonics", f"log power, degree {l}", np.log(h["power"][:, 0, l]))
    return pd.DataFrame(out)


def tv_table():
    """Train+Validation only, from the stored copy, so that the notebook opens without the raw feature files."""
    if (OUT / "features.csv").exists():
        return pd.read_csv(OUT / "features.csv")
    table = build_table()
    return table[table.split != "Test"].reset_index(drop=True)


def groups_of(columns):
    groups = {}
    for c in columns:
        if ": " in c:
            groups.setdefault(c.split(": ")[0], []).append(c)
    return groups


def linear():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def forest():
    return RandomForestClassifier(300, min_samples_leaf=3, random_state=0)


def splits(y, seed):
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def oof(X, y, seed, model=linear):
    """Out-of-fold score of one five-fold split."""
    p = np.zeros(len(y))
    for tr, te in splits(y, seed):
        m = model().fit(X[tr], y[tr])
        p[te] = m.decision_function(X[te]) if hasattr(m, "decision_function") else m.predict_proba(X[te])[:, 1]
    return p


def cv_auc(X, y, reps=20, model=linear):
    return float(np.mean([roc_auc_score(y, oof(X, y, s, model)) for s in range(reps)]))


def oof_mean(X, y, reps=20):
    return np.mean([oof(X, y, s) for s in range(reps)], 0)


def delta(a, b, y, n=2000):
    """Gain from score a to score b: point value and 95% bootstrap interval over lesions."""
    rng, gains = np.random.default_rng(0), []
    while len(gains) < n:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            gains.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return dict(before=roc_auc_score(y, a), after=roc_auc_score(y, b), gain=roc_auc_score(y, b) - roc_auc_score(y, a),
                low=float(np.percentile(gains, 2.5)), high=float(np.percentile(gains, 97.5)))


def forward(blocks, X, y, seed, max_steps=6):
    """Greedy forward selection of columns blocks by inner five-fold AUC. Stops when nothing helps."""
    chosen, best = [], 0.5
    for _ in range(max_steps):
        scores = {}
        for name, cols in blocks.items():
            if name not in chosen:
                pick = sum((blocks[c] for c in chosen), []) + cols
                scores[name] = roc_auc_score(y, oof(X[:, pick], y, seed + 1000))
        if not scores:
            break
        name = max(scores, key=scores.get)
        if scores[name] <= best + 1e-3:
            break
        chosen.append(name)
        best = scores[name]
    return chosen


def nested_selection(blocks, X, y, seed):
    """One outer five-fold split: selection sees only each training part. Returns out-of-fold scores and the choices."""
    p, choices = np.zeros(len(y)), []
    for tr, te in splits(y, seed):
        chosen = forward(blocks, X[tr], y[tr], seed)
        choices.append(chosen)
        pick = sum((blocks[c] for c in chosen), []) or [0]
        p[te] = linear().fit(X[tr][:, pick], y[tr]).decision_function(X[te][:, pick])
    return p, choices


def permuted_nested(blocks, X, y, seed):
    rng = np.random.default_rng(seed)
    z = rng.permutation(y)
    return roc_auc_score(z, nested_selection(blocks, X, z, seed)[0])


def main(reps=20, permutations=100):
    table = build_table()
    table = table[table.split != "Test"].reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "features.csv", index=False)
    y = table.y.values
    groups = groups_of(table.columns)
    columns = [c for g in groups.values() for c in g]
    X = table[columns].values.astype(float)
    index = {g: [columns.index(c) for c in cols] for g, cols in groups.items()}
    feature_blocks = {c: [i] for i, c in enumerate(columns)}
    size = index["Size"]
    pool = Parallel(n_jobs=-1)

    def after_size(cols):
        v = np.c_[np.ones(len(y)), X[:, size]]
        return X[:, cols] - v @ np.linalg.lstsq(v, X[:, cols], rcond=None)[0]

    rows = {}
    for g, cols in index.items():
        rows[g] = dict(n=len(cols), kind="spectral" if g in SPECTRAL else "classical", lr=cv_auc(X[:, cols], y, reps), forest=cv_auc(X[:, cols], y, 5, forest),
                       with_size=cv_auc(X[:, cols + size], y, reps) if g != "Size" else np.nan, after_size=cv_auc(after_size(cols), y, reps) if g != "Size" else np.nan)
    pd.DataFrame(rows).T.to_csv(OUT / "groups.csv")

    names = list(index)
    pairs = pool(delayed(cv_auc)(X[:, index[a] + index[b]], y, reps) for a, b in itertools.combinations(names, 2))
    pd.DataFrame([(a, b, s) for (a, b), s in zip(itertools.combinations(names, 2), pairs)], columns=["a", "b", "auc"]).to_csv(OUT / "pairs.csv", index=False)

    every = sum(index.values(), [])
    spectral = sum((index[g] for g in SPECTRAL), [])
    classical = sum((index[g] for g in CLASSICAL), [])
    sets = {"Everything": every, "All classical": classical, "All spectral": spectral, "Size + spiculation": size + index["Spiculation"],
            "Size + spiculation + elongation": size + index["Spiculation"] + index["Elongation and sphericity"],
            "Size + spiculation + tips": size + index["Spiculation"] + index["Tips"], "Size + spiculation + tips + elongation": size + index["Spiculation"] + index["Tips"] + index["Elongation and sphericity"]}
    pd.DataFrame({n: dict(lr=cv_auc(X[:, c], y, reps), forest=cv_auc(X[:, c], y, 5, forest), n=len(c)) for n, c in sets.items()}).T.to_csv(OUT / "sets.csv")

    # Leave one group out from everything, and add each group to the best classical base, with intervals on out-of-fold scores.
    base = sets["Size + spiculation"]
    full, base_scores = oof_mean(X[:, every], y, reps), oof_mean(X[:, base], y, reps)
    effects = {}
    for g, cols in index.items():
        effects[g] = dict(drop=delta(oof_mean(X[:, [c for c in every if c not in cols]], y, reps), full, y))
        if g not in ("Size", "Spiculation"):
            effects[g]["add_to_base"] = delta(base_scores, oof_mean(X[:, base + cols], y, reps), y)
    (OUT / "effects.json").write_text(json.dumps(effects, indent=1))

    # Selection inside cross-validation, by feature group and by single feature.
    report = {}
    for name, blocks in (("groups", index), ("features", feature_blocks)):
        done = pool(delayed(nested_selection)(blocks, X, y, seed) for seed in range(reps))
        aucs = [roc_auc_score(y, p) for p, _ in done]
        counts = {}
        for _, choices in done:
            for chosen in choices:
                for c in chosen:
                    counts[c] = counts.get(c, 0) + 1
        report[name] = dict(auc=float(np.mean(aucs)), sd=float(np.std(aucs)), folds=reps * 5, picked={k: v / (reps * 5) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])},
                            size=float(np.mean([len(c) for _, cs in done for c in cs])))
    report["groups"]["null"] = pool(delayed(permuted_nested)(index, X, y, seed) for seed in range(permutations))
    (OUT / "selection.json").write_text(json.dumps(report, indent=1))

    # Stability of sparse models over bootstraps, on standardised features.
    def sparse_run(seed):
        i = np.random.default_rng(seed).integers(0, len(y), len(y))
        m = make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", C=0.1, solver="liblinear")).fit(X[i], y[i])
        return m[-1].coef_[0]
    coef = np.array(pool(delayed(sparse_run)(s) for s in range(500)))
    pd.DataFrame(dict(feature=columns, selected=(coef != 0).mean(0), positive=(coef > 0).mean(0), mean=coef.mean(0))).sort_values("selected", ascending=False).to_csv(OUT / "stability.csv", index=False)


if __name__ == "__main__":
    main()
