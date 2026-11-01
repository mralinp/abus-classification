# Stability of the Laplace–Beltrami spectrum

Roadmap Phase 3, fourth step. Evidence and proofs: [`notebooks/lb-stability.ipynb`](../../notebooks/lb-stability.ipynb).

## Implementation

- `abus_classification/features/shape_descriptor/shape_descriptors.py`: `spectrum_bound`, the guaranteed range of eigenvalue ratios between two meshes with the same faces.
- `abus_classification/lb_stability.py`: size-matched surfaces of all 200 lesions, under simulated boundary noise (0.25, 0.5 and 1 mm, 3 draws), grids of `V^(1/3)/16` and `/32`, and 2 and 4 times the slice spacing. `--predicted` compares Zhou 2021's predicted masks with the true ones on the 70 Test patches. `--samples` saves meshes for the figures.
- `test/utils_tests/features_tests/test_signature.py`: the bound on four deformations, its value for rigid motion and scale, low modes under fine bumps, and a dumbbell neck.

Labels are used on Train+Validation only; Test is used for geometry. The choices table is in the notebook, Section 2.

## Requirements

TDSC-ABUS under `data/tdsc/` (or `TDSC_PATH`). For `--predicted`: `data/eswa2025.npz` and `data/zhou2021_checkpoints/zhou2021_seed{0,1,2}.pt`, the stored weights, which are only read; their checksums were unchanged after the run. No Kaggle kernel. On a 15-core Mac with Apple Silicon: main run 13.4 min, `--predicted` about 1 min of mask smoothing and meshing plus inference.

## Reproducing the results

From a checkout with the project environment in `.venv`:

```bash
.venv/bin/python -m pytest test/utils_tests/features_tests/test_signature.py -q
.venv/bin/python -m abus_classification.lb_stability
.venv/bin/python -m abus_classification.lb_stability --predicted
.venv/bin/python -m abus_classification.lb_stability --samples
```

Then run the notebook, which writes the `docs/assets/lb-3d-necks.png`, `lb-3d-noise.png`, `lb-3d-predicted.png`, `lb-icc.png` and `lb-predicted-scatter.png` figures. Executing it needs `nbclient` or Jupyter, which are not project dependencies.

## Expected results

| Check | Value |
|---|---|
| Zhou Dice, seed 0, against the stored value | 0.724 both |
| ICC of lambda * area at 0.25 / 0.5 / 1 mm noise, k = 1 | 0.971 / 0.812 / 0.533 |
| Same, k = 20 | 0.818 / 0.444 / 0.144 |
| Same, k = 100 | 0.571 / 0.132 / 0.038 |
| ICC of volume at the same noise | 0.998 / 0.989 / 0.973 |
| Spearman, true vs predicted, mean of seeds: volume, elongation | 0.920, 0.636 |
| Same, lambda * area at k = 1, 20, 100 | 0.402, -0.013, -0.210 |
| Train+Validation AUC of lambda_1 * area, clean and at 1 mm | 0.608, 0.589 |

## Status

The stored spectra come from the current code. After they were computed, `measure` was refactored without a full rerun; one lesion's clean spectrum matched its stored values to 7e-13. Not run: predictions on full volumes, calibrated annotation noise, the Test split's labels.
