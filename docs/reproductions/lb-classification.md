# Classification with every feature set

Evidence: [`notebooks/lb-classification.ipynb`](../../notebooks/lb-classification.ipynb). Train+Validation only; the Test split is untouched.

## Implementation

`abus_classification/lb_classification.py` joins the stored results of `lb-descriptors`, `lb-stability` and `lb-spharm` with Tan's features into 112 features in 13 groups. It writes `features.csv` without Test rows, and `tv_table` returns only those rows. It then runs:

- each group alone, with size and after removing size, and every pair of groups;
- named feature sets, leave-one-group-out, and each group added to size and spiculation, with bootstrap intervals;
- greedy forward selection nested inside repeated five-fold CV, for groups and for single features, with 100 label permutations of the group version;
- L1 stability selection over 500 bootstraps.

The notebook then fits the small model and explains single lesions. Test: `test_classification_table_excludes_test_and_selection_ignores_noise`.

## Requirements

`data/tan_features.csv`, and `results/` from the earlier Laplace–Beltrami steps. CPU only, no Kaggle kernel: it took about 2.7 min on a 15-core Mac. The notebook reads stored results and also needs `results/lb-roundness/sample_meshes.npz`.

## Reproducing the results

From a checkout with the project environment in `.venv`:

```bash
.venv/bin/python -m pytest test/utils_tests/features_tests/test_signature.py -q
.venv/bin/python -m abus_classification.lb_classification
```

Then run the notebook, which writes `docs/assets/lb-groups.png`, `lb-pairs.png`, `lb-selection.png` and `lb-3d-explanations.png`. Executing it needs `nbclient` or Jupyter, which are not project dependencies.

## Expected results

A second run gave identical `groups.csv` and `selection.json`.

| Check | Value |
|---|---|
| Nested group selection, AUC | 0.814 (SD 0.018) |
| Nested feature selection, AUC | 0.787 (SD 0.020) |
| Shuffled labels, nested groups: median, 95th percentile | 0.502, 0.582 |
| Spiculation dropped from everything, AUC lost | 0.071 (0.022 to 0.125) |
| Hersch ratio added to size and spiculation | +0.045 (0.006 to 0.089) |
| All classical, all spectral, everything | 0.837, 0.671, 0.805 |

## Status

The stored results come from the current code. Not run: any model on the Test split, and any of this on predicted masks.
