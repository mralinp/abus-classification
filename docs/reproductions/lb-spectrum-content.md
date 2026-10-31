# What the Laplace–Beltrami spectrum determines, and the 3D feature figures

Roadmap Phase 3, second open step. Evidence: [`notebooks/lb-spectrum-content.ipynb`](../../notebooks/lb-spectrum-content.ipynb). Illustrations of every feature set: [`notebooks/lb-features-3d.ipynb`](../../notebooks/lb-features-3d.ipynb). The 3D cell added to [`notebooks/lb-invariance.ipynb`](../../notebooks/lb-invariance.ipynb) uses the same sample meshes.

## Implementation

- `abus_classification/features/shape_descriptor/shape_descriptors.py`: `heat_trace_coefficients` fits area, Euler number and the curvature term of the heat trace.
- `abus_classification/lb_spectrum_content.py`: each of the 200 lesions at 0.4 mm, as staircase and Gaussian-smoothed surfaces (0.4 and 0.8 mm), largest mesh piece only, up to 300 eigenvalues. `--samples` saves meshes of 6 Train lesions at the 25th, 50th and 75th volume percentiles of each class.
- `abus_classification/utils/mesh.py`: `draw_mesh` draws a mesh in 3D with per-vertex colours.
- `test/utils_tests/features_tests/test_signature.py`: the heat fit on analytic and mesh spectra, and a Sunada pair.

Labels are used on Train+Validation only. The choices table is in `lb-spectrum-content.ipynb`, Section 3.

## Requirements

TDSC-ABUS under `data/tdsc/` (or `TDSC_PATH`). `results/lb-spectrum-content/tan_control.csv` holds three spiculation columns and two others copied from `data/tan_features.csv`, for the positive control. CPU only, no Kaggle kernel, run on a 15-core Mac. The experiment's run time was not recorded.

## Reproducing the results

From a checkout with the project environment in `.venv`:

```bash
.venv/bin/python -m pytest test/utils_tests/features_tests/test_signature.py -q
.venv/bin/python -m abus_classification.lb_spectrum_content
.venv/bin/python -m abus_classification.lb_spectrum_content --samples
```

Then run the three notebooks, which write `docs/assets/lb-3d-*.png` and `lb-weyl.png`. Executing them needs `nbclient` or Jupyter, which are not project dependencies.

## Expected results

| Check | Value |
|---|---|
| Median area from Weyl / true area, k = 100 | 1.05 staircase, 1.04 smoothed 0.4 mm, 1.04 smoothed 0.8 mm |
| Fitted Euler number on fine sphere / torus (true 2 / 0) | 1.98 / -0.07 |
| Fitted Euler number, genus-0 lesions, smoothed 0.8 mm, 5th to 95th percentile | -0.57 to 4.80 |
| Sunada pair, largest relative eigenvalue gap | 3e-15 |
| Five-fold CV AUC on Train+Validation, volume | 0.71 |
| Same, lambda * area, up to k = 100, after removing volume | 0.37–0.49 |
| Same, Tan spiculation, after removing volume | 0.69 |

## Status

The stored results come from the current code. The spectra were computed once; a repeat run was not compared. Not run: size-matched meshing or smoothing, non-linear models, the Test split.
