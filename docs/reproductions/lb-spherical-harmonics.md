# Spherical harmonics and the Laplace–Beltrami spectrum

Roadmap Phase 3, fifth step. Evidence and theory: [`notebooks/lb-spherical-harmonics.ipynb`](../../notebooks/lb-spherical-harmonics.ipynb).

## Implementation

- `abus_classification/features/shape_descriptor/spharm.py`: `sphere_map` (conformalized mean curvature flow, then Möbius balancing), `real_harmonics`, `spharm_coefficients`, `spharm_power`, `map_quality`.
- `abus_classification/lb_spharm.py`: for each of the 200 lesions, a size-matched genus-0 surface (smoothing raised 1.5× at a time until closed and genus 0), clean and at 0.25, 0.5 and 1 mm boundary noise, 2 draws each. It stores power up to degree 15, degree-1 axes, reconstruction error, map quality, lambda_1 and map energy. `--predicted` does the same for Zhou 2021's masks on the 70 Test patches.
- `abus_classification/lb_stability.py`: `surface` now takes the smoothing as an argument; its default keeps earlier results unchanged.
- `test/utils_tests/features_tests/test_signature.py`: map and reconstruction on a sphere, harmonics against the operator, rotation invariance, where lobes appear, and degree-1 axes.

Labels are used on Train+Validation only; Test patches are used for geometry. The choices table is in the notebook, Section 2.

## Requirements

TDSC-ABUS under `data/tdsc/` (or `TDSC_PATH`). For `--predicted`: `data/eswa2025.npz` and `data/zhou2021_checkpoints/zhou2021_seed{0,1,2}.pt`, only read; their checksums were unchanged after the run. The notebook also reads the stored results of `lb-stability`, `lb-roundness` and `lb-spectrum-content`. CPU only, no Kaggle kernel: on a 15-core Mac the main run took 7.1 min and `--predicted` 2.8 min.

## Reproducing the results

From a checkout with the project environment in `.venv`:

```bash
.venv/bin/python -m pytest test/utils_tests/features_tests/test_signature.py -q
.venv/bin/python -m abus_classification.lb_spharm
.venv/bin/python -m abus_classification.lb_spharm --predicted
```

Then run the notebook, which writes `docs/assets/lb-3d-harmonics.png`, `lb-3d-spheremap.png`, `lb-3d-reconstruction.png` and `lb-spharm-icc.png`. Executing it needs `nbclient` or Jupyter, which are not project dependencies.

## Expected results

| Check | Value |
|---|---|
| Lesions mapped; needing topology correction | 200; 19 |
| Largest share of flipped triangles | 0.0005 |
| Map energy / 8 pi: lowest, median, highest | 0.9995, 1.0016, 1.0492 |
| Lesions with lambda_1 * area at or below map energy | 100% |
| Median reconstruction error at degree 15, share of radius | 0.014 |
| ICC of power at 0.5 mm: degrees 1, 5, 10, 15 | 0.801, 0.789, 0.698, 0.584 |
| Spearman, true vs predicted: degrees 1, 3, 5, 10, 15 | 0.289, 0.537, 0.532, 0.475, 0.172 |
| Five-fold CV AUC after removing volume: power 2–5; spiculation | 0.558; 0.695 |

## Status

The stored results come from the current code, in one run; a repeat run was not compared. Not run: area-preserving maps, topology-preserving repair, the Test split's labels.
