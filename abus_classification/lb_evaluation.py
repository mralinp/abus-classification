"""Laplace-Beltrami features for classification: combinations, selection, ACC, AUC and the TDSC-ABUS benchmark.

Two stages, so that the Test split cannot influence any choice:

    python -m abus_classification.lb_evaluation --freeze     # Train+Validation only: compares, selects and saves the models
    python -m abus_classification.lb_evaluation --evaluate   # scores the frozen models once on Test; needs the frozen file

A rule picks the winner on Train+Validation: the highest mean of cross-validated AUC and ACC. Its two winners, over all
features and over LB features only, are the primary results, named before Test was read. Every other combination is
scored on Test too, and the best of those is reported as such, since picking it there would be biased.
"""
import itertools
import json
import os
import sys

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"   # one process per task

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from abus_classification import challenge
from abus_classification.lb_classification import (RESULTS, SPECTRAL, build_table, groups_of, nested_selection, oof, permuted_nested,
                                                   splits, tv_table)
from abus_classification.protocol import auc_ci

OUT = RESULTS / "lb-evaluation"
FROZEN = OUT / "frozen.json"
PICK_SHARE = 0.4        # a feature enters a model if nested selection chose it in this share of training folds


CLASSIFIERS = {
    "Logistic C=0.01": lambda: LogisticRegression(C=0.01, max_iter=3000), "Logistic C=0.1": lambda: LogisticRegression(C=0.1, max_iter=3000),
    "Logistic C=1": lambda: LogisticRegression(max_iter=3000), "Logistic L1 C=0.1": lambda: LogisticRegression(penalty="l1", C=0.1, solver="liblinear"),
    "Logistic L1 C=0.5": lambda: LogisticRegression(penalty="l1", C=0.5, solver="liblinear"), "LDA shrinkage": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    "Naive Bayes": GaussianNB, "SVM linear C=0.01": lambda: SVC(kernel="linear", C=0.01), "SVM linear C=0.1": lambda: SVC(kernel="linear", C=0.1),
    "SVM RBF C=1": lambda: SVC(C=1), "SVM RBF C=10": lambda: SVC(C=10), "kNN 5": lambda: KNeighborsClassifier(5), "kNN 15": lambda: KNeighborsClassifier(15),
    "Random forest": lambda: RandomForestClassifier(200, min_samples_leaf=3, random_state=0), "Extra trees": lambda: ExtraTreesClassifier(200, min_samples_leaf=3, random_state=0),
    "Gradient boosting": lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=0),
    "Small neural net": lambda: MLPClassifier((16,), alpha=1.0, max_iter=2000, random_state=0),
}


def make_model(g):
    """Scaling, an optional univariate filter fitted on training data only, and a classifier."""
    steps = [StandardScaler()] + ([SelectKBest(f_classif, k=g["k"])] if g["k"] else []) + [CLASSIFIERS[g["classifier"]]()]
    return make_pipeline(*steps)


def probability(model, X):
    """Probability of malignant; a margin goes through a sigmoid, so p >= 0.5 is the classifier's own decision."""
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else 1 / (1 + np.exp(-model.decision_function(X)))


def grid_scores(X, y, g, reps):
    aucs, accs = [], []
    for seed in range(reps):
        p = np.zeros(len(y))
        for tr, te in splits(y, seed):
            p[te] = probability(make_model(g).fit(X[tr], y[tr]), X[te])
        aucs.append(roc_auc_score(y, p)), accs.append(((p >= 0.5) == y).mean())
    return dict(AUC=float(np.mean(aucs)), ACC=float(np.mean(accs)))


def scores(X, y, reps=20):
    """Cross-validated AUC and accuracy at p >= 0.5, the benchmark's threshold."""
    runs = [oof(X, y, s) for s in range(reps)]
    return dict(AUC=float(np.mean([roc_auc_score(y, r) for r in runs])), ACC=float(np.mean([((r >= 0) == y).mean() for r in runs])))


def freeze(reps=20, permutations=100):
    assert not FROZEN.exists() or "--force" in sys.argv, "already frozen: changing models after a Test run would invalidate it"
    table = tv_table()
    assert len(table) == 130 and "Test" not in set(table.split)
    y = table.y.values
    groups = {g: c for g, c in groups_of(table.columns).items() if g in SPECTRAL}
    columns = [c for g in groups.values() for c in g]
    X = table[columns].values.astype(float)
    index = {g: [columns.index(c) for c in cols] for g, cols in groups.items()}
    single = {c: [i] for i, c in enumerate(columns)}
    pool = Parallel(n_jobs=-1)
    OUT.mkdir(parents=True, exist_ok=True)

    alone = {g: scores(X[:, c], y, reps) for g, c in index.items()}
    alone["All LB features"] = scores(X, y, reps)
    pd.DataFrame(alone).T.to_csv(OUT / "groups.csv")
    names = list(index)
    pair = pool(delayed(scores)(X[:, index[a] + index[b]], y, reps) for a, b in itertools.combinations(names, 2))
    pd.DataFrame([dict(a=a, b=b, **s) for (a, b), s in zip(itertools.combinations(names, 2), pair)]).to_csv(OUT / "pairs.csv", index=False)
    triple = pool(delayed(scores)(X[:, sum((index[g] for g in t), [])], y, reps) for t in itertools.combinations(names, 3))
    pd.DataFrame([dict(groups=" + ".join(t), **s) for t, s in zip(itertools.combinations(names, 3), triple)]).to_csv(OUT / "triples.csv", index=False)

    def nested(blocks):
        done = pool(delayed(nested_selection)(blocks, X, y, s) for s in range(reps))
        counts = {}
        for _, folds in done:
            for chosen in folds:
                for c in chosen:
                    counts[c] = counts.get(c, 0) + 1 / (reps * 5)
        return dict(AUC=float(np.mean([roc_auc_score(y, p) for p, _ in done])), ACC=float(np.mean([((p >= 0) == y).mean() for p, _ in done])),
                    size=float(np.mean([len(c) for _, f in done for c in f])), picked=dict(sorted(counts.items(), key=lambda kv: -kv[1])))

    selection = dict(groups=nested(index), features=nested(single))
    selection["groups"]["null"] = pool(delayed(permuted_nested)(index, X, y, s) for s in range(permutations))
    selection["features"]["null"] = pool(delayed(permuted_nested)(single, X, y, s) for s in range(permutations))
    (OUT / "selection.json").write_text(json.dumps(selection, indent=1))

    chosen_groups = [g for g, v in selection["groups"]["picked"].items() if v >= PICK_SHARE]
    chosen_features = [c for c, v in selection["features"]["picked"].items() if v >= PICK_SHARE]
    everything = groups_of(table.columns)
    size, spiculation = everything["Size"], everything["Spiculation"]
    classical = sum((everything[g] for g in everything if g not in SPECTRAL), [])
    sets = {"LB selected features": chosen_features, "LB selected groups": sum((groups[g] for g in chosen_groups), []), "All LB": columns,
            "Size + LB selected features": size + chosen_features, "Size + all LB": size + columns, "Size + spiculation": size + spiculation,
            "Size + spiculation + Hersch ratio": size + spiculation + groups["Hersch ratio"], "All classical": classical,
            "All classical + LB selected features": classical + chosen_features, "All classical + all LB": classical + columns}
    grid = [dict(set=n, classifier=c, k=k) for n, cols in sets.items() for c in CLASSIFIERS for k in (None, 5, 10, 20) if k is None or k < len(cols)]
    data = {n: table[cols].values.astype(float) for n, cols in sets.items()}
    results = pool(delayed(grid_scores)(data[g["set"]], y, g, 5) for g in grid)
    frame = pd.DataFrame([dict(g, k=g["k"] or 0, **r) for g, r in zip(grid, results)])
    frame.to_csv(OUT / "grid_cv.csv", index=False)

    # The whole search again, inside cross-validation: an unbiased view of what picking the winner achieves.
    lb_sets = [n for n in sets if n.startswith("LB") or n == "All LB"]
    outer = [(seed, fold, tr, te) for seed in range(2) for fold, (tr, te) in enumerate(splits(y, seed + 500))]
    honest = {"Any features": np.zeros((2, len(y))), "LB features only": np.zeros((2, len(y)))}
    inner = pool(delayed(grid_scores)(data[g["set"]][tr], y[tr], g, 1) for _, _, tr, _ in outer for g in grid)
    winners = []
    for n, (seed, fold, tr, te) in enumerate(outer):
        block = pd.DataFrame(inner[n * len(grid):(n + 1) * len(grid)]).assign(i=range(len(grid)))
        block["goal"] = (block.AUC + block.ACC) / 2
        for label, mask in (("Any features", np.ones(len(grid), bool)), ("LB features only", np.array([g["set"] in lb_sets for g in grid]))):
            g = grid[int(block[mask].sort_values("goal").i.iloc[-1])]
            honest[label][seed, te] = probability(make_model(g).fit(data[g["set"]][tr], y[tr]), data[g["set"]][te])
            winners.append(dict(label=label, **{k: g[k] or 0 if k == "k" else g[k] for k in g}))
    pd.DataFrame(winners).to_csv(OUT / "nested_winners.csv", index=False)
    nested_scores = {k: dict(AUC=float(np.mean([roc_auc_score(y, v[i]) for i in range(2)])), ACC=float(np.mean([((v[i] >= 0.5) == y).mean() for i in range(2)]))) for k, v in honest.items()}

    frame["goal"] = (frame.AUC + frame.ACC) / 2
    best = frame.sort_values("goal", ascending=False).iloc[0]
    best_lb = frame[frame.set.isin(lb_sets)].sort_values("goal", ascending=False).iloc[0]
    plain = {n: dict(set=n, classifier="Logistic C=1", k=None) for n in sets}
    named = {"Best overall by the rule": spec(best), "Best LB-only by the rule": spec(best_lb), **{"Logistic: " + n: g for n, g in plain.items()}}
    FROZEN.write_text(json.dumps(dict(primary=["Best overall by the rule", "Best LB-only by the rule"], sets=sets, grid=grid, named=named, nested=nested_scores), indent=1))


def spec(row):
    return dict(set=row.set, classifier=row.classifier, k=int(row.k) or None)


def evaluate():
    frozen = json.loads(FROZEN.read_text())
    assert not (OUT / "test_results.csv").exists() or "--force" in sys.argv, "Test has already been scored once"
    train, table = tv_table(), build_table()
    test = table[table.split == "Test"].reset_index(drop=True)
    assert len(test) == 70 and len(train) == 130
    y, y_train = test.y.values, train.y.values

    def predict(g):
        cols = frozen["sets"][g["set"]]
        return probability(make_model(g).fit(train[cols].values.astype(float), y_train), test[cols].values.astype(float))

    grid = pd.DataFrame([dict(g, k=g["k"] or 0, **challenge.classification(y, predict(g))) for g in frozen["grid"]])
    grid.to_csv(OUT / "grid_test.csv", index=False)
    probabilities = {n: predict(g) for n, g in frozen["named"].items()}
    pd.DataFrame(dict(case=test.case, y=y, **probabilities)).to_csv(OUT / "test_predictions.csv", index=False)
    rows = {}
    for name, p in probabilities.items():
        low, high = auc_ci(y, p)
        rows[name] = dict(frozen["named"][name], **challenge.classification(y, p), AUC_low=low, AUC_high=high, sensitivity=(p >= 0.5)[y == 1].mean(), specificity=(p < 0.5)[y == 0].mean())
    pd.DataFrame(rows).T.to_csv(OUT / "test_results.csv")
    bench = challenge.benchmark({n: challenge.classification(y, p) for n, p in probabilities.items()})
    bench[["ACC", "AUC", "Cls_Score", "Cls_Rank"]].to_csv(OUT / "benchmark.csv")


if __name__ == "__main__":
    evaluate() if "--evaluate" in sys.argv else freeze()
