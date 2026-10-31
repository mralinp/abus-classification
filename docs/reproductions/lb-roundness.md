# Roundness and lobulation from the Laplace–Beltrami spectrum

Roadmap Phase 3, third step. Evidence and proofs: [`notebooks/lb-roundness.ipynb`](../../notebooks/lb-roundness.ipynb).

## Implementation

- `abus_classification/lb_roundness.py`: each of the 200 lesions on its own grid, step (volume)^(1/3) / 24, Gaussian smoothing 0.04 and 0.08 (volume)^(1/3), largest mesh piece, up to 200 eigenvalues, plus area, volume, Euler number and principal-axis extents. `--samples` saves meshes of Train genus-0 lesions at the 5th–95th percentiles of lambda_1 * area, per class.
- `abus_classification/features/shape_descriptor/shape_descriptors.py`: `sphere_level_spread`.
- `test/utils_tests/features_tests/test_signature.py`: Hersch on spheres and ellipsoids, the selection rule against its first-order prediction, odd degrees at second order.

Labels are used on Train+Validation only. The choices table is in the notebook, Section 2.

## Requirements

TDSC-ABUS under `data/tdsc/` (or `TDSC_PATH`), and `results/lb-spectrum-content/tan_control.csv` for the positive control. CPU only, no Kaggle kernel: the experiment took 3.7–3.8 min on a 15-core Mac.

## Reproducing the results

From a checkout with the project environment in `.venv`:

```bash
.venv/bin/python -m pytest test/utils_tests/features_tests/test_signature.py -q
.venv/bin/python -m abus_classification.lb_roundness
.venv/bin/python -m abus_classification.lb_roundness --samples
```

Then run the notebook, which writes `docs/assets/lb-3d-lobes.png`, `lb-3d-roundness.png` and `lb-hersch.png`. Executing it needs `nbclient` or Jupyter, which are not project dependencies.

## Expected results

A second run matched the stored eigenvalues to about 5e-13. Values at the lighter smoothing:

| Check | Value |
|---|---|
| Largest absolute Spearman, lambda * area and volume, k up to 150 | 0.152 (0.209 at the heavier smoothing) |
| Hersch violations, genus-0 lesions | 0 of 186 |
| Largest and median lambda_1 * area / 8 pi | 0.883, 0.610 |
| Spearman, that ratio and elongation | -0.686 |
| Five-fold CV AUC on Train+Validation, volume | 0.708 |
| Same, lambda_1 * area: alone, with volume, after volume | 0.587, 0.717, 0.573 |
| Same, elongation: alone, with volume, after volume | 0.643, 0.746, 0.632 |
| Same, Tan spiculation after volume | 0.693 |

## Status

The stored results come from the current code. Not run: the Test split, stability under segmentation error, local HKS or WKS maps as classifier inputs.
