# Barekatrezaei et al. 2026: provisional reproduction record

## Status and scope

**Awaiting the first author's finalized source, including the CNN implementation.** Ali Naderiparizi
has implemented part of the work and is waiting for his research partner, the first author, to supply
the remaining finalized source. The CNN currently checked into this repository is a reconstruction
from the paper text. It is not the finalized author CNN and has not been verified against it.

Paper: S. Barekatrezaei, A. Naderiparizi, E. Kozegar, J. Ghofrani and M. Soryani,
*Breast mass classification in 3D ABUS based on Laplace-Beltrami spectra and dual path CNN*,
Expert Systems with Applications 299:129973 (2026),
[doi:10.1016/j.eswa.2025.129973](https://doi.org/10.1016/j.eswa.2025.129973).
Legacy `eswa2025` filenames and kernel names are retained to identify the existing artifacts.

The [study notebook](../../notebooks/2-barekatrezaei2026.ipynb) keeps the paper tables, our provisional
results, differences, ablations and spectral diagnostics. README performance comparisons are deferred
while implementation fidelity is unresolved. These records must not be read as a finalized assessment
of the paper, nor as original-author verification. Receiving source, reviewing it and rerunning
experiments are separate pending steps.

## Implementation and data flow

This account describes source inspected at repository revision
`8240f1f3c8baea7fc09787b7677d76814c2cd3eb`; that is an inspection baseline, not a verified training revision.

| Component | Current implementation | What it establishes / what is pending |
| --- | --- | --- |
| Feature cache | [lesion_cache.py](../../abus_classification/data/lesion_cache.py) | Builds one row per volume: radiomics, eigenvalues, VOI and mask; comparison with finalized feature code pending |
| CNN | [model.py](../../abus_classification/papers/barekatrezaei2026/model.py) | Our dual-path reconstruction and training loop; finalized author architecture/training source awaited |
| Ensemble and ablations | [pipeline.py](../../abus_classification/papers/barekatrezaei2026/pipeline.py) | SVM, Extra Trees, CNN, voting and two stackers; settings need author comparison |
| CLI | [paper module](../../abus_classification/papers/barekatrezaei2026/__main__.py) | Defaults to 60 epochs, seed 0; `--ablate` selects the ablation routine |
| Legacy entry point | [reproduce_eswa2025.py](../../reproduce_eswa2025.py) | Compatibility shim used by the existing kernels and notebook |
| Verification | [test_papers.py](../../test/papers_tests/test_papers.py) | CNN output shape and shared metric checks; does not verify author fidelity or convergence |

Data flows from the official Train, Validation and Test splits into `data/eswa2025.npz`, then to
five-fold out-of-fold predictions on Train + Validation and fold-averaged Test predictions. SVM,
Extra Trees and CNN probabilities feed voting or a stacker trained on out-of-fold predictions.
The target uses `y=1` for malignant, reversing the loader's malignant label `0`.

### Current choices to compare with the finalized source

These are source-level facts about our implementation. Unless confirmed against the paper or author
source, they must not be presented as verified paper requirements.

| Topic | Current choice | Reason / fidelity question |
| --- | --- | --- |
| Lesion location | Ground-truth masks determine features and lesion-centered patches | Makes classification possible without a detector; differs from an end-to-end challenge pipeline |
| Radiomics | Height/width ratio, entropy, mean intensity, margin contrast, intensity variance, sphericity; physical spacing used | Verify definitions and preprocessing against finalized feature code |
| Laplacian | Cotangent matrix on the lesion mesh; ordinary eigenproblem without a mass matrix; first 100 eigenvalues | Verify discretization and units. This is distinct from a generalized mass-matrix eigenproblem |
| Spectral preprocessing | Drop the first eigenvalue, clamp at `1e-12`, log10, standardize | Source comment motivates logs by orders of magnitude; verify transforms and eigenvalue indexing |
| Scaling scope | LB scaler fits all Train + Validation before folds; CNN radiomics scaler in the `both` variant does too | Held-out fold information enters scaling. A fold-isolated rerun is needed to assess its effect; no code change or rerun in this documentation job |
| Patches | `(128, 64, 80)` at 0.476 mm isotropic spacing; smoothing and linear resampling; intensity divided by 255; mask threshold 0.5 | Reconstructed preprocessing; verify geometry, interpolation and normalization |
| CNN paths | Separate VOI and mask paths; widths 16/32/64/128; two Conv3d–InstanceNorm–ReLU blocks per width, max pooling, global average pooling | Provisional architecture; finalized CNN code is the blocking reference |
| CNN head | Concatenated paths and full retained LB vector; dense 64, ReLU, dropout 0.5, scalar logit | Verify fusion, feature dimensions and head against finalized CNN |
| Training | Adam `1e-3`, batch 8, BCE-with-logits, cosine schedule, 60 epochs in kernels, paired random flips on two patch axes | Reconstruction choices, not a confirmed transcription of all original settings; no early stopping in this loop |
| Randomness | Torch seed set; flips use NumPy randomness without a local NumPy seed in `fit_cnn` | Seed labels alone do not establish exact reproducibility; preserve this limitation when comparing runs |
| Tabular models | Linear probabilistic SVM and 500-tree Extra Trees; fold-fitted scaling and Isomap with 5 neighbors to 2 dimensions by default | Verify model tuning, reducers and hyperparameters |
| Combination | Mean voting; HistGradientBoosting and logistic-regression stackers | Two alternative stackers were explored; seed 0 has only the original stacking row |

## Requirements and execution record

Real experiments require authorized access to the TDSC-ABUS Train, Validation and Test splits, a
Python environment with the repository's scientific dependencies and PyTorch, and a GPU for CNN
training. Local analysis of aggregate CSVs does not require the real volumes. Rebuilding the cache
does require volumes and masks. `TDSC_PATH` selects the split root before importing the cache module;
its default is `data/tdsc` in the repository.

Existing runs used Kaggle and private dataset sources `tdsc-abus2023-trainval` and
`tdsc-abus2023-test`. The checked-in runners clone `feat/barekatrezaei2026`; that branch is mutable.
Exact historical training commits, Kaggle run versions, GPU models and dependency snapshots are not
recorded in these CSVs. Recover them from run metadata before claiming a bit-for-bit rerun. Do not
substitute the current inspection baseline for that missing provenance.

| Kernel | Inputs and source-recorded invocation | Outputs and dependency order |
| --- | --- | --- |
| `mralinp/eswa2025-reproduction` ([runner](../../kaggle/reproduction/run.py)) | Finds the three split folders, links them under `/tmp/tdsc`, sets `TDSC_PATH`, calls `r.main(epochs=60)` | `results.csv`; copied `eswa2025.npz` feature cache; run before the next two kernels |
| `mralinp/eswa2025-ablations` ([runner](../../kaggle/ablations/run.py)) | Links the previous kernel's cache; calls `r.ablations(60, 0)` | `ablations.csv`; seed 0 ablation grid |
| `mralinp/eswa2025-seeds` ([runner](../../kaggle/seeds/run.py)) | Links the same cache; calls `r.main(60, s)` for seeds 1 and 2 | `results_seed1.csv`, `results_seed2.csv` |

The notebook and existing Kaggle guide record historical approximate costs: cache construction about
1.7 hours on four cores, ablations about three hours, additional seeds about two hours. These are
historical estimates, not timings measured by this documentation update or guarantees for a new run.
The notebook's approximate per-seed runtime likewise does not identify hardware or a kernel version.

For a future rerun, prepare the source and datasets, run `reproduction` first, attach its output to
`ablations` and `seeds`, then collect CSVs and predictions before opening the notebook. Consult the
[Kaggle guide](../../kaggle/README.md) for source-owner configuration. The current generic push helper
is not established here as the exact submission command used for these historical runs. No Kaggle
submission, cache build or GPU training is rerun by this documentation change.

### Artifacts retained and missing

The repository retains four aggregate tables in [results/eswa2025](../../results/eswa2025/):

- `table3_seed0.csv`: the initial seed-0 result; its `Stacking` row is treated as HGB by the notebook.
- `results_seed1.csv` and `results_seed2.csv`: include both `Stacking-HGB` and `Stacking-LR`.
- `ablations.csv`: seed-0 reducer, dimension, feature and base-model ablations.

The current main routine also writes `data/eswa2025_preds_s<seed>.npz` with `oof`, `tst`, `y_tv` and
`y_te`, but those files are not present in `results/eswa2025/`. The feature cache, CNN checkpoints and
full training histories are not retained there either. The CNN routine does not save checkpoints or
per-epoch histories. Aggregate tables cannot recover them. Document recovery from actual Kaggle
outputs if available; no verified artifact-download command is recorded here.

The notebook reads the stored CSVs and compares them to its transcribed paper table. Source inspection
and table loading can be repeated without training. New training commands derived from the current
source remain unverified rerun instructions until executed with recorded provenance.

## Provisional findings and interpretation

The paper table transcribed in the notebook reports stacking AUC 0.9350 and accuracy 0.8429, and voting
AUC 0.8725. Our stored voting AUCs are 0.8625, 0.8517 and 0.8633 for seeds 0, 1 and 2 respectively.
Stored HGB stacking AUCs are 0.7546, 0.6787 and 0.7383; LR AUCs are 0.8242 and 0.8225 for seeds 1 and 2.
These are historical outputs of our provisional reconstruction, not evidence of finalized CNN fidelity.

The ablation table does not reproduce the expected preference for the default Isomap/2-dimensional
setting. Importantly, `T7 no LB` removes LB from the **tabular** models; the CNN in that experiment
still receives LB. It is not a complete removal of LB from the ensemble. `T8 CNN none` removes the
CNN's extra features, while its tabular partners keep theirs. Interpret each intervention by its
actual input paths rather than its short table label.

The notebook retains an exploratory table comparing raw-log, area-normalized and ratio eigenvalue
features with logistic regression. It reports strong correlation between raw-log spectra and lesion
volume and weaker classification after normalization. Its generating script, split indices and raw
outputs are not retained in this record. Treat it as a diagnostic hypothesis about size dependence in
this implementation, not a general conclusion about Laplace–Beltrami descriptors. A reproducible,
fold-isolated study belongs in the later spectral investigation.

No training history here establishes CNN overfitting, its cause, or the need for early stopping in this
paper. Do not transfer the Yu 2024 overfitting observations to this study. No intervention resolving
the stacking gap has been established. The finalized CNN, feature definitions, transforms, fold
handling and ensemble details all remain candidates for comparison.

The notebook's challenge comparison averages aggregate ACC/AUC across the available seed rows. It is
not a seed-probability ensemble. LR has only two seeds. It is classification-only and uses ground-truth
masks; it cannot establish an overall challenge rank or replace a prediction-level evaluation.

## Author verification and next steps

Current source status: finalized source including the CNN is awaited from the first author. Feedback
verifying the checked-in CNN or the complete reconstruction is not recorded here. No new outreach is
sent by this documentation job.

When source is supplied:

1. Record the source version, license/permission and which components are finalized. Preserve this
   provisional implementation and its results as an identifiable baseline.
2. Compare architecture, feature definitions, Laplacian/eigenproblem, preprocessing, folds, training
   and stacking choices. Cite paper sections and source locations; ask focused questions where unclear.
3. Verify corrections with method-specific checks and rerun affected configurations with exact commit,
   kernel version, dependencies and hardware recorded. Retain per-case predictions and training histories.
4. Report original and corrected results separately. Record what changed, why, whether it explains the
   discrepancy and what remains unknown. Any diagnostic training change gets its own variant label.
5. Request author review of the precise implementation and experiment snapshot; record the reviewed
   scope and feedback before claiming verification. Update the README assessment only when supported.

This record and the notebook are the review package for continuing that comparison. Their detail is
intended to support collaboration and corrections, including findings that remain unresolved.
