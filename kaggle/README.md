# Kaggle GPU training for ABUS paper studies

Script kernels (GPU + internet). Push one with `python kaggle/push.py <name>`, which adds each source's owner from `~/.kaggle/owners.json` (see `push.py`).

Kaggle is the current GPU training platform; notebooks explain the studies and load stored results.
Use the [research index](../docs/papers.md) to find a paper's implementation, notebook and reproduction
record. Record the exact Git commit, Kaggle run/version, accelerator, dependencies, data, settings and
artifact locations for each run. The branch named in `run.py` can move; it is not sufficient provenance.
If these details are unavailable for an older run, mark them unknown rather than infer them.

Paper-default runs, diagnostic ablations and changed training procedures must have distinct names and
outputs. A run responding to overfitting must record the observation, proposed cause, intervention,
changed data split and outcome in the paper's reproduction record. See the
[research workflow](../docs/research-workflow.md).

## Barekatrezaei 2026 (legacy ESWA 2025 paths)

Awaiting the first author's finalized source, including the CNN implementation. The existing CNN is
our reconstruction; these kernels and their outputs document provisional experiments. See the
[reproduction record](../docs/reproductions/barekatrezaei2026.md) before reusing their results.

- `reproduction/`: clones `feat/barekatrezaei2026`, finds the TDSC-ABUS splits in the private datasets `tdsc-abus2023-trainval` and `-test`, builds the feature cache (about 1.7 h), runs Table 3. Its output holds `eswa2025.npz`.
- `ablations/`, `seeds/`: reuse that cache through `kernel_sources` (no rebuild). Ablations (paper Tables 4-8, seed 0) take about 3 h; seeds 1 and 2 about 2 h.

Results are in `results/eswa2025/`. Kernels run under the `mralinp` account. Kernels that read another kernel's output through `kernel_sources` need that kernel run first under the same account: `reproduction/` before the others, and `zhou2021/` before `zhou2021-eval/`.

## Zhou 2021

- `zhou2021/`: trains CMSVNetIter (3 seeds x 100 epochs, about 4 h). Its output holds the trained checkpoints.
- `zhou2021-eval/`: re-evaluates those checkpoints, read as an input, without training.
- `zhou2021-ablations/`, `zhou2021-ablations2/`: held-out diagnostics and the paper's Table 1 models (seed 0).

Results are in `results/zhou2021/`.

## Wang 2020

`wang2020/`: builds the patches from the private datasets, then trains CNN A, CNN B, the single views and CNN A from scratch (3 seeds each). Finished runs are never trained again. Results are in `results/wang2020/`.

## Yu 2024

`yu2024/`: runs on the `mralinp` account. Builds the slices from the private datasets, trains ResNet34 on whole slices, lesion crops and from scratch (3 seeds each), and repeats the paper's five-fold CV on Train. Finished runs are never trained again. Results are in `results/yu2024/`.

`yu2024-early-stopping/`: the same trainings with early stopping and an LR scheduler on the official Validation split, one training per GPU on T4 x 2. Push with `--accelerator NvidiaTeslaT4`. It needs the Kaggle secret `RCLONE_CONF` attached (a base64-encoded rclone config with a `gdrive` remote, `scope=drive.file`), never committed. Start it from Kaggle's editor with Save & Run All, since a CLI push could not read the secret. Output and checkpoints go to `abus-checkpoints/yu2024-early-stopping` on Drive every 10 minutes, and a restarted kernel resumes from them.
