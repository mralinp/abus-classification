# Spectral descriptors as classifiers

Roadmap Phase 3, spectral descriptors row. Evidence and theory: [`notebooks/lb-descriptors.ipynb`](../../notebooks/lb-descriptors.ipynb).

## Implementation

- `abus_classification/features/shape_descriptor/shape_descriptors.py`: `spectral_summaries` gives scale-free, pose-free Shape-DNA, HKS, WKS, GPS and tips summaries; helpers `weighted_quantiles` and `local_maxima`.
- `abus_classification/lb_descriptors.py`: those summaries on size-matched surfaces of all 200 lesions, clean and at 0.25, 0.5 and 1 mm boundary noise, 2 draws each. `--predicted` does the same for Zhou 2021's masks on the 70 Test patches.
- `test/utils_tests/features_tests/test_signature.py`: rotation and scale, mean HKS against the heat trace, mean WKS, tips on a bumpy sphere, and the Sunada pair.

Labels are used on Train+Validation only; Test patches are used for geometry. The choices table is in the notebook, Section 2.

## Requirements

TDSC-ABUS under `data/tdsc/` (or `TDSC_PATH`). For `--predicted`: `data/eswa2025.npz` and `data/zhou2021_checkpoints/zhou2021_seed{0,1,2}.pt`, only read; their checksums were unchanged after the run. The notebook also reads stored results of `lb-stability`, `lb-roundness` and `lb-spectrum-content`. CPU only, no Kaggle kernel: on a 15-core Mac the main run took 6.7 min and `--predicted` 1.2 min. The notebook's 200-permutation test adds a few minutes.

## Reproducing the results

From a checkout with the project environment in `.venv`:

```bash
.venv/bin/python -m pytest test/utils_tests/features_tests/test_signature.py -q
.venv/bin/python -m abus_classification.lb_descriptors
.venv/bin/python -m abus_classification.lb_descriptors --predicted
```

Then run the notebook, which writes `docs/assets/lb-3d-tips.png`, `lb-3d-wks-band.png` and `lb-descriptors-signal.png`. Executing it needs `nbclient` or Jupyter, which are not project dependencies.

## Expected results

Five-fold CV AUC on Train+Validation, repeated 20 times:

| Features | Alone | With volume | After volume |
|---|---|---|---|
| Volume | 0.708 | | |
| Shape-DNA, k up to 20 | 0.508 | 0.607 | 0.444 |
| HKS mean (heat trace) | 0.602 | 0.718 | 0.564 |
| WKS 95th percentile | 0.660 | 0.759 | 0.635 |
| Tips | 0.675 | 0.766 | 0.638 |
| Tan spiculation | 0.788 | 0.802 | 0.693 |

Gain from adding tips to volume: 0.063, 95% interval -0.015 to 0.145. Best gain over 9 families: 0.049 with 3 CV repeats, permutation p = 0.020 from 200 shuffles of the features.

## Status

The stored results come from the current code, in one run; a repeat run was not compared. The Test split's labels have not been used. Not run: a final model on Test, models on the signature maps themselves.
