# Laplace–Beltrami invariance

Roadmap Phase 3, first open step. Evidence and proofs: [`notebooks/lb-invariance.ipynb`](../../notebooks/lb-invariance.ipynb).

## Implementation

- `abus_classification/features/shape_descriptor/shape_descriptors.py`: `lb_spectrum`, the eigenvalues of `L phi = lambda M phi`, multiplied by area by default.
- `abus_classification/lb_invariance.py`: for each of the 200 lesions, 5 seeded random rigid motions, re-voxelised at 0.4 mm and meshed. Motion 0 is repeated at 0.8 mm. Each render is meshed as a staircase surface and after Gaussian smoothing (sigma 0.8 mm); the largest mesh piece is kept. It stores the eigenvalues with and without `M`, and the area, volume, vertex count, Euler number and piece count.
- `test/utils_tests/features_tests/test_signature.py`: rigid motion, scale, an exact isometric dent, a stretch, `eig(L) = lambda * area / n` on icospheres, and a moved voxel ball.

Labels are used on Train+Validation only, so Test stays unused while descriptors are designed. The choices table is in the notebook, Section 2.

## Requirements

TDSC-ABUS under `data/tdsc/` (or `TDSC_PATH`) and the ESWA cache `data/eswa2025.npz`, the output of the `eswa2025-reproduction` kernel. CPU only, no Kaggle kernel: it ran in 2.6–3.6 min on a 15-core Mac.

## Reproducing the results

From a checkout with the project environment in `.venv`:

```bash
.venv/bin/python -m pytest test/utils_tests/features_tests/test_signature.py -q
.venv/bin/python -m abus_classification.lb_invariance
```

This writes `results/lb-invariance/spectra.npz`. Then run `notebooks/lb-invariance.ipynb`, which also writes `docs/assets/lb-invariance.png`. The notebook needs `nbclient` or Jupyter, which are not project dependencies.

## Expected results

Two runs gave eigenvalues matching to about 4e-12 and identical areas and vertex counts. No seed variation applies beyond the fixed motions.

| Check | Value |
|---|---|
| Median variation of lambda * area across motions | 0.2% smoothed, 1.0% staircase |
| Median `eig(L) * n / (lambda * area)`, k = 1–20 | 0.994–1.000 |
| Median lambda * area, 0.8 mm / 0.4 mm | 0.98–1.00 smoothed, 0.99–1.04 staircase |
| Median `eig(L)`, 0.8 mm / 0.4 mm | about 4.0 smoothed, 4.7 staircase |
| Spearman, stored ESWA k = 99 and native vertex count | -1.000 |
| Five-fold CV AUC on Train+Validation, 20 repeats | volume 0.710, lambda * area 0.591, both 0.652, stored ESWA 0.723 |

## Status

The stored results come from the current code. Not run: modes above k = 20, other smoothing levels, the Test split, and lambda * area inside the full ESWA ensemble.
