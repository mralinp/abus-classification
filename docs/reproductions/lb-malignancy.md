# What malignancy means, and what a 3D surface can show

Roadmap Phase 3, first part of "Theory-driven descriptors". Evidence: [`notebooks/lb-malignancy.ipynb`](../../notebooks/lb-malignancy.ipynb). Designed on Train+Validation, then scored once on Test from a plan frozen in advance. `lb-feature-evaluation` had already used Test, so this is a second look at that split.

## Implementation

`abus_classification/features/shape_descriptor/malignancy_signs.py` holds the descriptors:

- **Operator on the coordinates:** `mesh_spectrum`, `spectral_moments`, `spectral_roundness`, `area_budget`, `mean_curvature`.
- **Lobes:** `surface_height` above the surface's own smooth body, and `surface_peaks`, by persistence.
- **Orientation:** `beam_orientation`, the depth share and tilt of the low modes against the beam.
- **Tissue lines:** `line_field` (Tan's detector), `radial_lines`, `line_summary`, `line_surface_field`, `line_surface_summary`.

`abus_classification/lb_malignancy.py` has these stages:

- no flag: features for all 200 lesions, with simulated boundary noise, into `features.npz`. Geometry and image only.
- `--control`: the line excess out to 30 mm, on sham lesions moved elsewhere in the same scan, and for dark lines, into `control.npz`.
- `--predicted`: the same descriptors on Zhou 2021's masks for Test patches, into `predicted.npz`. Test labels are not used.
- `--analyse`: Train+Validation only: single features, families, models, the three-feature hybrid (`frozen.json`), reliability and the tables the notebook reads.
- `--samples`: meshes, fields and coronal planes of six Train lesions for the figures.
- `--freeze`: Train+Validation only: 12 named models with their coefficients, single-feature directions and volume lines, and five readings fixed in advance, into `test_plan.json`.
- `--evaluate`: applies those stored numbers once to Test, refits nothing, and refuses to run twice without `--force`. It writes the `test_*` files.

Tests: 15 in `test/utils_tests/features_tests/test_signature.py`, from `test_the_cotangent_laplacian_spends_exactly_twice_the_area_on_any_surface` onward, with hand-computed values.

## Requirements

`data/tdsc/`, `data/tan_features.csv`, `data/eswa2025.npz` and `data/zhou2021_checkpoints/` for the predicted masks, and `results/` from the earlier Laplace–Beltrami steps. CPU only, except that the predicted masks use MPS if available. No Kaggle kernel. On a 15-core Mac: features about 11 min, control and predicted masks about 7 min together, analysis about 3 min.

The notebook reads stored results. Executing it needs `nbclient` and `nbformat`, which are not project dependencies.

## Reproducing the results

From a checkout with the project environment in `.venv`:

```bash
.venv/bin/python -m pytest test/utils_tests/features_tests/test_signature.py -q
.venv/bin/python -m abus_classification.lb_malignancy
.venv/bin/python -m abus_classification.lb_malignancy --control
.venv/bin/python -m abus_classification.lb_malignancy --predicted
.venv/bin/python -m abus_classification.lb_malignancy --analyse
.venv/bin/python -m abus_classification.lb_malignancy --samples
.venv/bin/python -m abus_classification.lb_malignancy --freeze --force
.venv/bin/python -m abus_classification.lb_malignancy --evaluate --force
```

`--force` overwrites stored results. The original order was freeze first, then one evaluation. Rerunning reproduces the numbers; it does not make Test unseen.

The notebook writes `docs/assets/lb-malignancy-*.png`.

## Expected results

`test_plan.json` SHA-256, recorded before Test was read: `e3a3827fa22a0508544466c248f13f6cb9e3fed05930d997419bf62e6aff0e2c`. It holds only for the original run, as it can change with library versions.

**Frozen models on Test, 70 lesions, true masks:**

| Model | ACC | AUC | 95% CI | Board rank | Cross-validated on Train+Validation: AUC, ACC |
|---|---|---|---|---|---|
| **Hybrid: size, radial lines, tilt (primary)** | 0.814 | 0.854 | 0.758–0.934 | 1 of 9 | 0.835, 0.759 |
| Size + spiculation | 0.871 | 0.889 | 0.792–0.969 | 1 of 9 | 0.795, 0.737 |
| Size + radial lines | 0.786 | 0.902 | 0.826–0.959 | 1 of 9 | 0.803, 0.713 |
| Size + all designed | 0.786 | 0.899 | 0.822–0.960 | 1 of 9 | 0.814, 0.731 |
| Hybrid + fine budget | 0.757 | 0.888 | 0.808–0.950 | 2 of 9 | 0.835, 0.749 |
| Size + spiculation + tilt | 0.843 | 0.860 | 0.760–0.938 | 1 of 9 | Not computed |
| Radial lines alone | 0.729 | 0.850 | 0.751–0.931 | 2 of 9 | 0.763, 0.689 |
| Size + surface signs | 0.743 | 0.848 | 0.757–0.932 | 2 of 9 | 0.765, 0.706 |
| Spiculation alone | 0.800 | 0.840 | 0.729–0.929 | 1 of 9 | 0.766, 0.696 |
| Hybrid, lines over 4–30 mm | 0.771 | 0.834 | 0.732–0.921 | 2 of 9 | 0.847, 0.782 |
| Size | 0.714 | 0.758 | 0.635–0.868 | 5 of 9 | 0.707, 0.684 |
| Surface signs alone | 0.643 | 0.733 | 0.608–0.849 | 5 of 9 | 0.634, 0.599 |

The hybrid's features were narrowed on Train+Validation. The procedure that picks them inside each training fold scores AUC 0.820, ACC 0.738 there.

**Readings fixed in advance:**

| Question | Result | Reading |
|---|---|---|
| Hybrid better than size + spiculation? | −0.035 (−0.117 to 0.039) | No |
| Fine area budget useful on the hybrid? | 0.034 (−0.009 to 0.079) | Not established |
| Line signal replicated? | AUC 0.775 (0.659–0.883); after volume 0.767 (0.649–0.878) | Yes |
| Reach replicated, malignant above benign in every shell from 4 to 30 mm? | All seven shells | Yes |
| Sham control passed, lesion minus sham AUC? | 0.336 (0.142–0.525); 57 lesions | Yes |

**Single features on Test, with Train+Validation directions and volume lines:** tilt AUC 0.488 (0.343–0.633), after volume 0.607 (0.461–0.743), so it failed. SP90 0.836, after volume 0.756. Height-to-width 0.636, after volume 0.670. Fine area budget 0.764, after volume 0.758 (0.625–0.873).

**Zhou 2021's masks, patches centred on the true lesion, three seeds:** hybrid AUC 0.820 (0.792–0.838), ACC 0.781 (0.771–0.786); size 0.697, 0.662. Post hoc, not in the plan: size + radial lines 0.835, 0.781; surface signs alone 0.640, 0.433.

**Also post hoc, from stored predictions:** size + radial lines against size + spiculation 0.013 (−0.042 to 0.078); dropping tilt from the hybrid 0.048 (0.005 to 0.101); lines added to size 0.145 (0.042 to 0.258); surface signs added to size 0.090 (−0.028 to 0.213).

**On Train+Validation:** radial lines AUC after volume 0.736, against 0.707 for SP90; Spearman with volume 0.04, against 0.40. Sham control, 99 lesions: AUC 0.78 at the lesion, 0.58 at the sham (0.47–0.70). True against predicted masks, rank: radial lines 0.80–0.84, depth share 0.72, tilt 0.43, surface signs 0.00–0.28.

## Status

The stored results come from the current code, from one freeze and one evaluation. **Test has now been used twice,** so further tuning against it would be biased. Size + radial lines was the best of 12 on this split, and it needs new data, such as IIPL-3D-ABUS. Exploratory and added after the first Train+Validation results: the sham control, the 30 mm profile, dark lines, wider shells and the narrowing to three features. Not run: spiculation on predicted masks, the fine area budget added to size + spiculation on held-out data, a full pipeline that finds its own lesions, an external dataset, fusion with intensity and CNN features.
