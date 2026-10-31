# Classification of Breast Cancer Lesions in 3D Automated Breast Ultrasound

**[Project page](https://alinaderiparizi.com/abus-classification/) · [Code](https://github.com/mralinp/abus-classification) · [Roadmap](#roadmap) · [Results so far](#results-so-far)**

Automated 3D breast ultrasound (ABUS) images the whole breast in a single sweep, but reading the
resulting volumes is slow, and telling malignant lesions from benign ones remains hard.

In previous work [7] we showed that the **Laplace–Beltrami spectrum of a lesion's surface** is an
effective feature for classifying breast masses in 3D ABUS. That paper established *that* it works,
not *why*. This project supplies the missing mathematics, toward a follow-up paper:

1. **Reproduce the strongest published results** on the public TDSC-ABUS 2023 dataset under one
   evaluation protocol, so every comparison is fair.
2. **Develop the mathematical foundations** of Laplace–Beltrami shape representation for tumors: what
   the spectrum provably captures about a lesion's shape, what it cannot, how stable it is, and how it
   relates to spherical harmonic decompositions.
3. **Derive better descriptors from that theory** and test whether they improve on the baselines.

The work is carried out at the Iran Image Processing Lab (IIPL) by Ali Naderiparizi and Sepideh Barekatrezaei,
under the supervision of Dr. Ehsan Kozegar and Dr. Mohsen Soryani (main supervisor). Code is released under
the MIT license.

![Five malignant and five benign lesions from TDSC-ABUS, each shown in the transversal plane through its centre at a common millimetre scale](docs/assets/sample-tumors-transversal.png)

---

## Results so far

All results below use the **official TDSC-ABUS test split** (70 lesions: 40 malignant, 30 benign).
Always answering "malignant" scores 57.1% accuracy on it.

| Method | Features / model | Test AUC | Test accuracy | Notes |
| --- | --- | --- | --- | --- |
| Tan et al. 2012, reproduced here | 11 radiomic features, RBF SVM | **0.86** (95% CI 0.77–0.94) | **78.6%** | CV AUC 0.83 ± 0.03 |
| Tan et al. 2012 without spiculation, reproduced here | 8 features, RBF SVM | 0.79 | 75.7% | CV AUC 0.74 |
| Tan et al. 2013, reproduced here | 14 features, LDA | 0.82 | 72.9% | CV AUC 0.82 |
| Barekatrezaei et al. 2026, reproduced here | Laplace–Beltrami spectra + dual-path CNN, voting | 0.85–0.86 | 71.4–77.1% | 3 seeds; stacking reaches only 0.68–0.82, see below |
| Zhou et al. 2021, reproduced here | Multi-task V-Net, CMSVNetIter | 0.67 (95% CI 0.53–0.79) | 62.9% | mean of 3 seeds; Dice 0.73 |
| Wang et al. 2020, reproduced here | Multiview Inception-v3 (CNN A), official boxes | 0.755 (95% CI 0.63–0.86) | 65.7% | mean of 3 seeds; transverse view alone reaches 0.80 |
| Yu et al. 2024, reproduced here | ResNet34 on lesion crops, soft voting | 0.81 (mean of seeds 0.825, 95% CI 0.72–0.92) | 70.0% | 3 seeds; whole slices reach 0.75 |
| TDSC-ABUS challenge winner (T1, SZU) | 3D ResNet | 0.889 | 75.7% | reported in the challenge paper |
| Barekatrezaei et al. 2026 — our previous work [7] | Laplace–Beltrami spectra + dual-path CNN | 0.935 | 84.3% | reported in [7] |

> **Comparing across rows.** The reproductions here compute features from the ground-truth lesion
> masks, and the CNN patches are centred on those masks. Challenge entries had to locate and segment lesions themselves, so their task was harder.
> Closing that gap — running every method on predicted segmentations as well — is part of Phase 2.

**What the Tan et al. reproductions show so far** (details in
[`1-tao2013.ipynb`](https://github.com/mralinp/abus-classification/blob/main/notebooks/1-tao2013.ipynb)):

- **Coronal spiculation replicates.** Tan's spiculation features score AUC 0.80–0.81 on their own,
  matching the 0.81–0.83 reported on their Nijmegen data, and removing them costs the classifier
  0.07 test AUC. They are not a proxy for lesion size.
- **Several classic ultrasound features do not transfer.** Posterior acoustic behavior falls to chance
  (AUC 0.50, against 0.80 in the paper), and the height-to-width ratio and margin contrast lose most of
  their signal.
- **Shape carries most of the signal on this data.** Among general radiomic features, boundary fractal
  dimension (AUC 0.75), surface area, compactness and sphericity (0.74) lead, while intensity features
  sit near chance — which motivates the shape-based approach in Phase 3.

| Tan's features on two lesions | ROC on the test split |
| --- | --- |
| ![Coronal plane, coronal spiculation map and transversal plane with the posterior region for a malignant and a benign lesion](docs/assets/tan-features-example.png) | ![ROC curves on the TDSC-ABUS test split for the reproduced Tan 2012 and Tan 2013 classifiers](docs/assets/tan-roc-test-split.png) |

**Barekatrezaei et al. 2026** (details in
[`2-barekatrezaei2026.ipynb`](https://github.com/mralinp/abus-classification/blob/main/notebooks/2-barekatrezaei2026.ipynb);
remaining work in [#28](https://github.com/mralinp/abus-classification/issues/28)):

- **Stacking does not reproduce yet.** The paper's stacked ensemble reports AUC 0.935; ours reaches 0.68–0.82,
  depending on seed and stacker. Voting (0.85–0.86) is close to the paper's 0.873.
- **The ablations do not reproduce either.** Removing the Laplace–Beltrami features does not hurt here, and the raw
  eigenvalues act mostly as a proxy for lesion size. We are waiting for the original code to compare the pipelines.

**Zhou et al. 2021** (details in
[`3-zhou2021.ipynb`](https://github.com/mralinp/abus-classification/blob/main/notebooks/3-zhou2021.ipynb)):

- **Below the paper, but not clearly so.** The paper reports AUC 0.787, accuracy 74.1% and Dice 0.778 from four-fold
  cross-validation on private data. Ours is AUC 0.67, accuracy 62.9% and Dice 0.73, and the paper's AUC lies inside our
  confidence interval.
- **Refinement helps segmentation, as in the paper.** The paper's classification gain does not show here: on one seed,
  the full method scores AUC 0.65 against 0.74 without refinement, with overlapping confidence intervals.
- **The classification head under-fits.** Its loss barely falls during training and training AUC only reaches about
  0.75. The cause is not yet tested.

**Wang et al. 2020** (details in
[`4-wang2020.ipynb`](https://github.com/mralinp/abus-classification/blob/main/notebooks/4-wang2020.ipynb)):

- **Far below the paper.** The paper reports AUC 0.947 with five-fold cross-validation on private data; ours is 0.755, and the
  paper's value is above our confidence interval.
- **The multiview gain does not show.** The transverse view alone scores highest (0.80), and ImageNet weights give no visible
  gain; all confidence intervals overlap.
- **Validation overstates the result.** Best validation AUC averages 0.94 for CNN A on 30 lesions, against 0.74 on test.

**Yu et al. 2024** (details in
[`6-yu2024.ipynb`](https://github.com/mralinp/abus-classification/blob/main/notebooks/6-yu2024.ipynb)):

- **Far below the paper, on its own data and protocol.** The paper reports soft voting AUC 0.986 from five-fold cross-validation on
  the TDSC Train split; the same protocol gives 0.85 here. The paper's numbers are also inconsistent: its voting rows list AUC equal
  to accuracy, and its hard voting sensitivity and specificity imply accuracy 0.937, not 0.907.
- **Soft voting helps only a little over hard voting.** It adds about 0.02 AUC, without a paired test yet.
- **Training memorises quickly.** Training accuracy reaches about 100%, by epoch 2 on whole slices. The likely cause is volumes
  being memorised; this is not yet tested.
- **Early stopping did not help, so the paper defaults stay the main result.** We added it because of that memorising, stopping on the
  official Validation split. Crops stay at ensemble AUC 0.825, whole slices fall from 0.765 to 0.676, and no pretraining rises from 0.687
  to 0.705; all paired 95% intervals include zero. Five-fold CV also falls, from 0.846 to 0.815 on crops. Validation AUC reaches
  0.91–0.995 against 0.69–0.80 on Test, so 30 validation volumes are a poor guide.
- **Challenge benchmark, classification only:** crops rank 4 of 9 (Cls_Score 0.734) with defaults and 3 of 9 (0.767) with early stopping.
  The AUC is equal, and the gap is one test volume at the 0.5 threshold. This method uses ground-truth masks, so it has no overall rank.

**Laplace–Beltrami invariance** (details in
[`lb-invariance.ipynb`](https://github.com/mralinp/abus-classification/blob/main/notebooks/lb-invariance.ipynb)):

- **The invariances hold.** Rigid motion, isometry and scale (λₖ·Area) are exact on meshes. Random rigid motions of re-voxelised lesions
  change λₖ·Area by about 0.2% on smoothed surfaces and 1% on staircase surfaces.
- **The eigenvalues in our previous work measure size.** They come from the cotangent matrix L without the mass matrix, and eig(L) ≈ λ·Area / n,
  with n the vertex count (ratio 0.99–1.00 here). The stored values rank lesions almost exactly by vertex count (Spearman −1.00 at k = 99), so they
  also depend on the voxel grid. In 29 of 200 lesions, an extra mesh piece shifts their indices.
- **Little signal remains in the low modes once size is removed.** On Train + Validation, cross-validated AUC is 0.59 for λₖ·Area with k = 1–20, 0.71 for
  volume and 0.72 for the stored values. Adding these modes to volume lowers it to 0.65. Higher modes and lighter smoothing are untested.

![Stored ESWA eigenvalues against mesh vertex count, eig(L) times vertex count against lambda times area, and raw against area-normalised eigenvalues by lesion volume](docs/assets/lb-invariance.png)

| Predicted and true tumors, Zhou et al. 2021 | Ablations against the paper |
| --- | --- |
| ![Coronal planes of five test lesions from worst to best Dice, with predicted and ground-truth contours](docs/assets/zhou-segmentation-overlays.png) | ![Bar charts of classification AUC and segmentation Dice for each ablation model, ours against the paper](docs/assets/zhou-ablations.png) |

---

## Roadmap

Status: ✅ done · 🔄 in progress · ⬜ planned

### Phase 0 — Foundations ✅

| | Task | Where |
| --- | --- | --- |
| ✅ | Reproducible environment with [uv](https://docs.astral.sh/uv/) | `pyproject.toml`, `uv.lock` |
| ✅ | TDSC-ABUS data loading | [`tdsc-abus2023-pytorch`](https://github.com/mralinp/tdsc-abus2023-pytorch) |
| ✅ | Physical voxel spacing and ABUS plane geometry (transversal, coronal, sagittal) | `abus_classification/utils/spacing.py` |
| ✅ | Dataset statistics and visualization, in millimetres | [`0-visualization.ipynb`](https://github.com/mralinp/abus-classification/blob/main/notebooks/0-visualization.ipynb) |
| ✅ | Tested feature library (radiomic, Tan et al., spectral shape descriptors) | `abus_classification/features`, `test/` |

### Phase 1 — Reproduce classical radiomics ✅

| | Paper | Authors | Approach and PR |
| --- | --- | --- | --- |
| ✅ | *[Computer-aided lesion diagnosis in automated 3-D breast ultrasound using coronal spiculation](https://doi.org/10.1109/TMI.2012.2184549)* [1, 3] | Tan et al. 2012 | Coronal spiculation after the mammography stellate-lesion statistic [4], 11 features, SVM; AUC 0.86. [`1-tao2013.ipynb`](https://github.com/mralinp/abus-classification/blob/main/notebooks/1-tao2013.ipynb), PR [#21](https://github.com/mralinp/abus-classification/pull/21) |
| ✅ | *[Evaluation of the effect of computer-aided classification of benign and malignant lesions on reader performance in automated three-dimensional breast ultrasound](https://doi.org/10.1016/j.acra.2013.07.013)* [2, 3] | Tan et al. 2013 | 14 features, LDA; AUC 0.82. [`1-tao2013.ipynb`](https://github.com/mralinp/abus-classification/blob/main/notebooks/1-tao2013.ipynb), PR [#21](https://github.com/mralinp/abus-classification/pull/21) |

### Phase 2 — Reproduce the strongest published results 🔄

Every method is re-run under one protocol: train on Train + Validation (130 lesions), report AUC with
a bootstrap confidence interval and accuracy on the official Test split, and compare pairs with
DeLong's test. Each is evaluated twice — from ground-truth masks and from predicted segmentations.
Each also reports the [challenge benchmark](notebooks/challenge-benchmark.ipynb) on the published leaderboard
(`abus_classification/challenge.py`). So far that covers classification only ([#34](https://github.com/mralinp/abus-classification/issues/34)).

| | Paper | Authors | Approach and PR |
| --- | --- | --- | --- |
| 🔄 | *[Breast mass classification in 3D ABUS based on Laplace-Beltrami spectra and dual path CNN](https://doi.org/10.1016/j.eswa.2025.129973)* [7], our previous work | Barekatrezaei et al. 2026 | Laplace–Beltrami spectra + dual-path CNN, stacked ensemble; voting AUC 0.85–0.86, stacking does not reproduce yet ([#28](https://github.com/mralinp/abus-classification/issues/28)). [`2-barekatrezaei2026.ipynb`](https://github.com/mralinp/abus-classification/blob/main/notebooks/2-barekatrezaei2026.ipynb), PR [#27](https://github.com/mralinp/abus-classification/pull/27) |
| ⬜ | *[Tumor detection, segmentation and classification challenge on automated 3D breast ultrasound: the TDSC-ABUS challenge](https://arxiv.org/abs/2501.15588)* [5], team T1 (SZU) | Luo et al. 2025 | 3D ResNet, U-Net and anchor-free detector; not started ([#30](https://github.com/mralinp/abus-classification/issues/30)) |
| ⬜ | *[Tumor detection, segmentation and classification challenge on automated 3D breast ultrasound: the TDSC-ABUS challenge](https://arxiv.org/abs/2501.15588)* [5], team T2 (POSTECH) | Luo et al. 2025 | Multi-task residual nnU-Net; not started ([#31](https://github.com/mralinp/abus-classification/issues/31)) |
| ⬜ | *[Tumor detection, segmentation and classification challenge on automated 3D breast ultrasound: the TDSC-ABUS challenge](https://arxiv.org/abs/2501.15588)* [5], team T3 (HU) | Luo et al. 2025 | nnU-Net, nnDetection and patch-based 3D ResNet-18; not started ([#32](https://github.com/mralinp/abus-classification/issues/32)) |
| ⬜ | *[Tumor detection, segmentation and classification challenge on automated 3D breast ultrasound: the TDSC-ABUS challenge](https://arxiv.org/abs/2501.15588)* [5], team T4 (KTH) | Luo et al. 2025 | STU-Net and 2.5D DenseNet201 with voting; not started ([#33](https://github.com/mralinp/abus-classification/issues/33)) |
| ✅ | *[Multi-task learning for segmentation and classification of tumors in 3D automated breast ultrasound images](https://www.sciencedirect.com/science/article/abs/pii/S1361841520302826)* [6] | Zhou et al. 2021 | Multi-task V-Net: joint segmentation and classification; test AUC 0.67, Dice 0.73 (paper, on its own data: 0.787, 0.778), with ablations; patches centred on ground-truth masks. [`3-zhou2021.ipynb`](https://github.com/mralinp/abus-classification/blob/main/notebooks/3-zhou2021.ipynb), PR [#26](https://github.com/mralinp/abus-classification/pull/26) |
| ✅ | *[Three-dimensional automated breast ultrasound (ABUS) tumor classification using a 2D-input network: soft voting or hard voting?](https://doi.org/10.3390/app142411611)* [8] | Yu et al. 2024 | ResNet34 on 2D slices with hard and soft voting; test AUC 0.81 from lesion crops, 0.75 from whole slices (paper: 0.986, five-fold CV on Train; ours on that protocol: 0.85). Early stopping did not improve it; challenge classification rank 4 of 9. [`6-yu2024.ipynb`](https://github.com/mralinp/abus-classification/blob/main/notebooks/6-yu2024.ipynb), PR [#43](https://github.com/mralinp/abus-classification/pull/43) |
| ✅ | *[Breast cancer classification in automated breast ultrasound using multiview convolutional neural network with transfer learning](https://doi.org/10.1016/j.ultrasmedbio.2020.01.001)* | Wang et al. 2020 | Multiview Inception-v3; test AUC 0.755 (paper, on its own data: 0.947), with view ablations; uses the official boxes. [`4-wang2020.ipynb`](https://github.com/mralinp/abus-classification/blob/main/notebooks/4-wang2020.ipynb), PR [#38](https://github.com/mralinp/abus-classification/pull/38) |
| 🔄 | *[Fully automatic classification of automated breast ultrasound (ABUS) imaging according to BI-RADS using a deep convolutional neural network](https://doi.org/10.1007/s00330-022-08558-0)* | Hejduk et al. 2022 | Three step-wise 2D CNNs and a sliding window over slices; running on Kaggle, no PR yet |
| ⬜ | Unified comparison table and statistical tests across all reproduced methods | — | — |

### Phase 3 — Mathematical foundations of Laplace–Beltrami shape representation ⬜

Our previous work [7] showed that Laplace–Beltrami spectra of the lesion surface separate malignant
from benign lesions well, but did not explain why. This phase supplies the mathematics: which
properties of a tumor's shape the spectrum provably captures, which it cannot, how stable it is under
the imperfections of real ultrasound segmentations, and how it relates to spherical harmonic
decompositions. Every claim is paired with an empirical check on TDSC-ABUS, and the theory is used to
derive better-founded descriptors.

| | Step | Notes |
| --- | --- | --- |
| ✅ | Watertight lesion surfaces | `utils.mesh.lesion_mesh`: meshes in millimetres; padding and no morphological closing keep all 200 surfaces closed |
| ✅ | Discrete Laplace–Beltrami operator | Cotangent scheme with lumped mass matrix, verified against the analytic spectrum of the unit sphere; convergence to the smooth operator depends on mesh conditions studied in [15, 16] |
| ✅ | Invariance | Proved and verified for rigid motion, isometry and scale (λₖ·Area). On re-voxelised lesions it holds to 0.2–1%. [`lb-invariance.ipynb`](https://github.com/mralinp/abus-classification/blob/main/notebooks/lb-invariance.ipynb) |
| ⬜ | What the spectrum determines | Surface area from Weyl's law [17]; area and Euler characteristic from the heat-trace expansion [18]; and its limits, the question posed by Kac [19] |
| ⬜ | Roundness and irregularity | Hersch's inequality, λ₁·Area ≤ 8π on genus-0 surfaces with equality only for the round sphere [20], makes the first eigenvalue a provable measure of departure from roundness; relate higher eigenvalues to lobulation and spiculation |
| ⬜ | Stability | Bound how far eigenvalues move under surface perturbation, mesh resolution, voxel anisotropy and segmentation error, and measure it on TDSC-ABUS |
| 🔄 | Spectral descriptors | Shape-DNA [9], heat kernel signature [10], wave kernel signature [11], global point signature [12] — implemented, classification pending |
| ⬜ | Spherical harmonics | Spherical harmonics are the Laplace–Beltrami eigenfunctions of the sphere, so a spherical harmonic decomposition [13, 14] and the lesion's own spectrum are two views of one framework; it needs genus-0 surfaces, so voxel-derived surfaces with handles require topology correction first |
| ⬜ | Theory-driven descriptors | Design descriptors from the results above, fuse them with spiculation, intensity and CNN features, and evaluate under the Phase 2 protocol with ablations |

![Lesion surfaces reconstructed by marching cubes for a malignant and a benign lesion, drawn at true physical proportions](docs/assets/lesion-surfaces-3d.png)

### Phase 4 — Paper ⬜

| | Task |
| --- | --- |
| ⬜ | Final experiments, ablations and statistical comparison |
| ⬜ | Figures and manuscript |
| ⬜ | Release of code, trained models and evaluation scripts |

---

## Datasets

### TDSC-ABUS 2023

The public dataset of the [TDSC-ABUS 2023 challenge](https://tdsc-abus2023.grand-challenge.org/): 200
ABUS volumes acquired with a GE Invenia ABUS at Harbin Medical University Cancer Hospital, each with a
voxel-level tumor mask and a malignant/benign label.

| Split | Volumes | Malignant | Benign |
| --- | --- | --- | --- |
| Train | 100 | 58 | 42 |
| Validation | 30 | 17 | 13 |
| Test | 70 | 40 | 30 |
| **Total** | **200** | **115** | **85** |

Access requires a signed data use agreement; see the
[dataset page](https://tdsc-abus2023.grand-challenge.org/Dataset/).

**Voxels are not cubes.** The volumes are 843–865 × 546–682 × 270–354 voxels at 0.200 mm along the
transducer, 0.073 mm in depth and 0.476 mm between slices, and the NRRD files do not record this. All
features and figures here use physical units; measured in voxels, lesions appear up to five times
thinner along one axis than they are.

### IIPL-3D-ABUS

A private dataset of 70 ABUS volumes (55 malignant, 15 benign) collected at the Iran Image Processing
Lab and segmented under the supervision of two expert radiologists. It cannot be shared; for enquiries,
contact <me@alinaderiparizi.com>.

---

## Getting started

### Install

The project is managed with [uv](https://docs.astral.sh/uv/), which also installs a suitable Python:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/mralinp/abus-classification.git
cd abus-classification
uv sync
```

`uv.lock` currently resolves for macOS, where PyTorch uses the Apple Silicon `mps` backend. To run on
Linux with CUDA, remove `environments` under `[tool.uv]` in `pyproject.toml` and run `uv lock`.

### Data

Place the TDSC-ABUS splits under `data/tdsc/`:

```text
data/tdsc/
├── Train/        DATA/, MASK/, labels.csv, bbx_labels.csv
├── Validation/
└── Test/
```

```python
from tdsc_abus2023_pytorch import TDSC, DataSplits
from abus_classification.features import tan

dataset = TDSC(path="data/tdsc", split=DataSplits.TRAIN, download=True)
volume, mask, label, bbox = dataset[0]          # label: 0 = malignant, 1 = benign

features = tan.extract_tan_features(volume, mask)   # millimetre-aware, uses TDSC-ABUS spacing
```

Passing `download=True` makes the dataset automatically downloaded if its not present on the system.

### Run

```bash
uv run pytest test/features_tests test/utils_tests test/papers_tests   # library and paper tests
uv run jupyter lab                                   # after: uv add --group notebooks jupyterlab
```

Training runs on Kaggle GPU kernels (see [`kaggle/README.md`](kaggle/README.md)). Notebooks load the stored
outputs in `results/`, so they open without a long run. To run a reproduction yourself:

```bash
uv run python -m abus_classification.papers.zhou2021 --epochs 100
```

### Google Colab

```python
!git clone https://github.com/mralinp/abus-classification.git
%cd abus-classification
!pip install -e .
```

---

## Repository layout

| Path | Contents |
| --- | --- |
| `abus_classification/papers/` | One subpackage per reproduced paper (`barekatrezaei2026`, `zhou2021`) |
| `abus_classification/protocol.py` | Shared evaluation protocol: device, metrics, folds, bootstrap CI |
| `abus_classification/lb_invariance.py` | Invariance experiment for the Laplace–Beltrami spectrum on TDSC-ABUS lesions |
| `abus_classification/challenge.py` | TDSC-ABUS 2023 challenge evaluation (Dice, HD, ACC, AUC, FROC) and published leaderboard |
| `abus_classification/data/` | Lesion cache: patches, radiomic and Laplace–Beltrami features per volume |
| `abus_classification/features/tan/` | Tan et al. (2012, 2013) features, including coronal spiculation |
| `abus_classification/features/radiology/` | General radiomic features in physical units |
| `abus_classification/features/shape_descriptor/` | Laplace–Beltrami operator, spectral signatures, boundary signatures |
| `abus_classification/features/texture/` | Grey-level co-occurrence texture features |
| `abus_classification/models/` | U-Net, V-Net, SE-FNet, multi-task V-Net |
| `abus_classification/utils/` | Meshing, voxel spacing and resampling, image utilities |
| `notebooks/` | Experiments — start with `0-visualization.ipynb` and `1-tao2013.ipynb`; one notebook per reproduced paper |
| `kaggle/` | Kaggle kernel scripts for the training runs |
| `results/` | Stored results of those runs, per paper |
| `test/` | Unit tests |

---

## References

1. T. Tan, B. Platel, H. Huisman, C. I. Sánchez, R. Mus, N. Karssemeijer. Computer-aided lesion diagnosis in automated 3-D breast ultrasound using coronal spiculation. *IEEE Transactions on Medical Imaging* 31(5):1034–1042, 2012. [doi:10.1109/TMI.2012.2184549](https://doi.org/10.1109/TMI.2012.2184549)
2. T. Tan, B. Platel, T. Twellmann, G. van Schie, R. Mus, A. Grivegnée, R. M. Mann, N. Karssemeijer. Evaluation of the effect of computer-aided classification of benign and malignant lesions on reader performance in automated three-dimensional breast ultrasound. *Academic Radiology* 20(11):1381–1388, 2013. [doi:10.1016/j.acra.2013.07.013](https://doi.org/10.1016/j.acra.2013.07.013)
3. T. Tan. *Automated 3D breast ultrasound image analysis.* PhD thesis, Radboud University Nijmegen, 2014. [pdf](https://repository.ubn.ru.nl/bitstream/handle/2066/121931/121931.pdf)
4. N. Karssemeijer, G. M. te Brake. Detection of stellate distortions in mammograms. *IEEE Transactions on Medical Imaging* 15(5):611–619, 1996. [doi:10.1109/42.538938](https://doi.org/10.1109/42.538938)
5. G. Luo, M. Xu, H. Chen, X. Liang, X. Tao, D. Ni, et al. Tumor detection, segmentation and classification challenge on automated 3D breast ultrasound: the TDSC-ABUS challenge. arXiv:2501.15588, 2025. [arXiv](https://arxiv.org/abs/2501.15588)
6. Y. Zhou, H. Chen, Y. Li, Q. Liu, X. Xu, S. Wang, D. Shen, P.-T. Yap. Multi-task learning for segmentation and classification of tumors in 3D automated breast ultrasound images. *Medical Image Analysis* 70:101918, 2021. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1361841520302826)
7. S. Barekatrezaei, A. Naderiparizi, E. Kozegar, J. Ghofrani, M. Soryani. Breast mass classification in 3D ABUS based on Laplace-Beltrami spectra and dual path CNN. *Expert Systems with Applications* 299:129973, 2026. [doi:10.1016/j.eswa.2025.129973](https://doi.org/10.1016/j.eswa.2025.129973)
8. S. Yu, X. Liang, S. Zhao, Y. Xie, Q. Sun. Three-dimensional automated breast ultrasound (ABUS) tumor classification using a 2D-input network: soft voting or hard voting? *Applied Sciences* 14(24):11611, 2024. [doi:10.3390/app142411611](https://doi.org/10.3390/app142411611)
9. M. Reuter, F.-E. Wolter, N. Peinecke. Laplace–Beltrami spectra as "Shape-DNA" of surfaces and solids. *Computer-Aided Design* 38(4):342–366, 2006. [doi:10.1016/j.cad.2005.10.011](https://doi.org/10.1016/j.cad.2005.10.011)
10. J. Sun, M. Ovsjanikov, L. Guibas. A concise and provably informative multi-scale signature based on heat diffusion. *Computer Graphics Forum* 28(5):1383–1392, 2009. [doi:10.1111/j.1467-8659.2009.01515.x](https://doi.org/10.1111/j.1467-8659.2009.01515.x)
11. M. Aubry, U. Schlickewei, D. Cremers. The wave kernel signature: a quantum mechanical approach to shape analysis. *IEEE International Conference on Computer Vision Workshops*, 2011.
12. R. M. Rustamov. Laplace-Beltrami eigenfunctions for deformation invariant shape representation. *Eurographics Symposium on Geometry Processing*, 2007.
13. C. Brechbühler, G. Gerig, O. Kübler. Parametrization of closed surfaces for 3-D shape description. *Computer Vision and Image Understanding* 61(2):154–170, 1995. [doi:10.1006/cviu.1995.1013](https://doi.org/10.1006/cviu.1995.1013)
14. M. Kazhdan, T. Funkhouser, S. Rusinkiewicz. Rotation invariant spherical harmonic representation of 3D shape descriptors. *Eurographics Symposium on Geometry Processing*, 2003.
15. K. Hildebrandt, K. Polthier, M. Wardetzky. On the convergence of metric and geometric properties of polyhedral surfaces. *Geometriae Dedicata* 123(1):89–112, 2006. [doi:10.1007/s10711-006-9109-5](https://doi.org/10.1007/s10711-006-9109-5)
16. A. I. Bobenko, B. A. Springborn. A discrete Laplace–Beltrami operator for simplicial surfaces. *Discrete & Computational Geometry* 38(4):740–756, 2007. [doi:10.1007/s00454-007-9006-1](https://doi.org/10.1007/s00454-007-9006-1)
17. H. Weyl. Das asymptotische Verteilungsgesetz der Eigenwerte linearer partieller Differentialgleichungen. *Mathematische Annalen* 71(4):441–479, 1912. [doi:10.1007/BF01456804](https://doi.org/10.1007/BF01456804)
18. H. P. McKean, I. M. Singer. Curvature and the eigenvalues of the Laplacian. *Journal of Differential Geometry* 1(1–2):43–69, 1967. [doi:10.4310/jdg/1214427880](https://doi.org/10.4310/jdg/1214427880)
19. M. Kac. Can one hear the shape of a drum? *The American Mathematical Monthly* 73(4):1–23, 1966. [doi:10.1080/00029890.1966.11970915](https://doi.org/10.1080/00029890.1966.11970915)
20. J. Hersch. Quatre propriétés isopérimétriques de membranes sphériques homogènes. *Comptes Rendus de l'Académie des Sciences de Paris* 270:1645–1648, 1970.

---

## Citation

A paper is in preparation. Until then, please cite this repository:

```bibtex
@misc{naderiparizi_abus_classification,
  author       = {Ali Naderiparizi and Sepideh Barekatrezaei and Ehsan Kozegar and Mohsen Soryani},
  title        = {Classification of Breast Cancer Lesions in 3D Automated Breast Ultrasound},
  year         = {2026},
  howpublished = {\url{https://github.com/mralinp/abus-classification}}
}
```

## License and contact

Code: [MIT License](LICENSE). The TDSC-ABUS data is subject to the challenge's data use agreement.
Questions and collaboration: <me@alinaderiparizi.com>
