# Research index

This index connects the current ABUS paper studies to their evidence. See the
[research workflow](research-workflow.md) for selection, implementation, evaluation and author review,
and the [README roadmap](../README.md#roadmap) for planned methods and Phase 3.

A stored result means an experiment exists. It does not by itself establish exact reproduction,
original-author verification, or equivalence to a challenge team's end-to-end pipeline.

## Current studies

| Study | Implementation | Notebook | Kaggle kernels | Stored results | Reproduction record / current limitation |
| --- | --- | --- | --- | --- | --- |
| Tan 2012 / Tan 2013 | [Tan features](../abus_classification/features/tan/) and [radiomics](../abus_classification/features/radiology/) | [1-tao2013](../notebooks/1-tao2013.ipynb) | No dedicated kernel in this checkout; classical fitting is in the notebook | Results/figures in the notebook | Standalone guide pending; one notebook currently covers both papers |
| Barekatrezaei 2026 | [Paper module](../abus_classification/papers/barekatrezaei2026/) | [2-barekatrezaei2026](../notebooks/2-barekatrezaei2026.ipynb) | [reproduction](../kaggle/reproduction/), [ablations](../kaggle/ablations/), [seeds](../kaggle/seeds/) | [eswa2025](../results/eswa2025/) | [Record](reproductions/barekatrezaei2026.md): finalized author source, including CNN code, awaited; existing runs provisional |
| Zhou 2021 | [Paper module](../abus_classification/papers/zhou2021/) | [3-zhou2021](../notebooks/3-zhou2021.ipynb) | [training](../kaggle/zhou2021/), [evaluation](../kaggle/zhou2021-eval/), [ablations](../kaggle/zhou2021-ablations/), [additional ablations](../kaggle/zhou2021-ablations2/) | [zhou2021](../results/zhou2021/) | Standalone guide pending; notebook documents implementation and experiments |
| Wang 2020 | [Paper module](../abus_classification/papers/wang2020/) | [4-wang2020](../notebooks/4-wang2020.ipynb) | [wang2020](../kaggle/wang2020/) | [wang2020](../results/wang2020/) | Standalone guide pending; notebook documents implementation and experiments |
| Yu 2024 | [Paper module](../abus_classification/papers/yu2024/) | [6-yu2024](../notebooks/6-yu2024.ipynb) | [paper defaults](../kaggle/yu2024/), [early-stopping variant](../kaggle/yu2024-early-stopping/) | [yu2024](../results/yu2024/) | [Record](reproductions/yu2024.md): paper-default and intervention runs, assumptions and unrun variants |

Original-author verification is not established for these implementations by the checked-in records.
The first paper's legacy `eswa2025` names refer to Barekatrezaei 2026; keep them for artifact traceability.
Hejduk 2022 and challenge-team methods are tracked in the roadmap, but their implementation artifacts
are not present in this checkout. The roadmap marks T1 (SZU) in progress; its implementation and Kaggle
run are under way. These running statuses are not independently verified by this documentation update.

## Shared benchmark and tools

- [Dataset visualization](../notebooks/0-visualization.ipynb) introduces the data and anatomical views.
- [Challenge benchmark notebook](../notebooks/challenge-benchmark.ipynb) explains scoring and limitations;
  [challenge.py](../abus_classification/challenge.py) implements it.
- [protocol.py](../abus_classification/protocol.py) supplies shared metrics, folds and GPU job scheduling.
- [Feature modules](../abus_classification/features/), [meshing and spacing tools](../abus_classification/utils/)
  and the [lesion cache](../abus_classification/data/lesion_cache.py) support multiple studies.
- [tdsc-abus2023-pytorch](https://github.com/mralinp/tdsc-abus2023-pytorch) supplies the dataset loader.

## Phase 3: current foundational study

[Invariance notebook](../notebooks/lb-invariance.ipynb),
[reproduction record](reproductions/lb-invariance.md),
[implementation](../abus_classification/lb_invariance.py),
[stored spectra](../results/lb-invariance/spectra.npz) and
[figure](assets/lb-invariance.png) record rigid-motion, isometry and scale investigations.
This foundational work has started alongside the benchmark studies. It does not establish completion
of the full Phase 3 classification comparison. Its retained study record describes CPU execution,
Train+Validation diagnostics and the unrun Test evaluation.

## Historical thesis and exploratory notebooks

The `notebooks/` directory also contains earlier work on boundary signatures, FFTs, texture, neural
networks and leave-one-out evaluation. These are research history and possible sources of reusable
ideas. Their presence does not establish compatibility with the current TDSC benchmark protocol.

Examples include [2D signatures](../notebooks/2d-signature.ipynb),
[3D FFT signatures](../notebooks/3d-fft-signature.ipynb),
[texture SVM](../notebooks/texture_svm.ipynb),
[leave-one-out](../notebooks/leave-one-out.ipynb), and
[multi-task networks](../notebooks/multi-task-segmentation-classification-3d.ipynb).
The [Laplace–Beltrami exploration](../notebooks/laplace-beltrami.ipynb) is background work, not a
completed Phase 3 classification study. Individual notebook provenance still needs review before
assigning it a definitive thesis/current-study status or reusing its scores in benchmark comparisons.

No historical files are moved or deleted by this documentation alignment.
