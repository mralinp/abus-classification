# Laplace–Beltrami features: combinations, selection, ACC, AUC and the TDSC-ABUS benchmark

Roadmap Phase 3, step after spherical harmonics. Evidence: [`notebooks/lb-feature-evaluation.ipynb`](../../notebooks/lb-feature-evaluation.ipynb).

## Implementation

`abus_classification/lb_evaluation.py` has two stages:

- `--freeze`, Train+Validation only: compares LB groups, pairs and triples, runs nested selection with 100 label permutations, then scores 510 combinations of 17 classifiers and 10 feature sets with optional univariate filters. A rule picks the highest mean of cross-validated AUC and ACC, over all features and over LB features only. It writes `frozen.json`.
- `--evaluate`: refits each combination on the 130 lesions, scores it on the 70 Test lesions, and writes `test_results.csv`, `grid_test.csv`, `test_predictions.csv` and `benchmark.csv`. It refuses to run twice without `--force`.

Features come from `abus_classification/lb_classification.py`, on ground-truth masks. The benchmark is `abus_classification/challenge.py`, classification only.

## Requirements

`data/tan_features.csv`, and `results/` from the earlier Laplace–Beltrami steps. CPU only, no Kaggle kernel, on a 15-core Mac. A smaller first freeze took about 5 min; the run of the full freeze finished within 10 min, and its exact time was not recorded.

## Reproducing the results

From a checkout with the project environment in `.venv`:

```bash
.venv/bin/python -m pytest test/utils_tests/features_tests/test_signature.py -q
.venv/bin/python -m abus_classification.lb_evaluation --freeze --force
.venv/bin/python -m abus_classification.lb_evaluation --evaluate --force
```

`--force` overwrites stored results. The original order was freeze first, then one evaluation. Rerunning only reproduces the stored numbers; it does not make Test unseen again. Then run the notebook, which writes `docs/assets/lb-evaluation-board.png`, `lb-evaluation-roc.png`, `lb-3d-classes.png` and `lb-class-overlap.png`. Executing it needs `nbclient` or Jupyter, which are not project dependencies.

## Expected results

`frozen.json` SHA-256, recorded before Test was read: `02c488e3c9f2937d3d25c7cdcb1437516f7d50c577fc4ce6dc31d46ce5231dde`. It holds only for the original run, as it can change with library versions.

| Frozen model | ACC | AUC | AUC 95% CI | Challenge classification rank |
|---|---|---|---|---|
| Rule's winner, LB only | 0.600 | 0.692 | 0.565–0.812 | 6 of 9 |
| Rule's winner, all features | 0.786 | 0.848 | 0.743–0.924 | 2 of 9 |
| Size + spiculation | 0.871 | 0.889 | 0.792–0.969 | 1 of 9 |
| Size + spiculation + Hersch ratio | 0.829 | 0.869 | 0.769–0.946 | 1 of 9 |

Search inside cross-validation on Train+Validation: all features AUC 0.803, ACC 0.746; LB only AUC 0.682, ACC 0.654. The "LB selected" list was chosen on all 130 lesions, so these are slightly generous.

## Status

The stored results come from the current code, from one freeze and one evaluation. A first, smaller freeze was replaced before Test was read, when more classifiers and classical features were requested. **Test has now been used once**, so further tuning against it would be biased. Not run: predicted masks, models on signature maps, an external dataset. Ranks use ground-truth masks and are not like-for-like with challenge teams.
