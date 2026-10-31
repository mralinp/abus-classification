# Kaggle kernels for the ESWA 2025 reproduction

Script kernels (GPU + internet). Push one with `python kaggle/push.py <name>`, which adds each source's owner from `~/.kaggle/owners.json` (see `push.py`).

- `reproduction/`: clones this branch, finds the TDSC-ABUS splits in the private datasets `tdsc-abus2023-trainval` and `-test`, builds the feature cache (about 1.7 h), runs Table 3. Its output holds `eswa2025.npz`.
- `ablations/`, `seeds/`: reuse that cache through `kernel_sources` (no rebuild). Ablations (paper Tables 4-8, seed 0) take about 3 h; seeds 1 and 2 about 2 h.

Results are in `results/eswa2025/`. Kernels run under the `mralinp` account. Kernels that read another kernel's output through `kernel_sources` need that kernel run first under the same account: `reproduction/` before the others, and `zhou2021/` before `zhou2021-eval/`. Edit the kernel `id`s if you push under another Kaggle account.

## Zhou 2021

- `zhou2021/`: trains CMSVNetIter (3 seeds x 100 epochs, about 4 h). Its output holds the trained checkpoints.
- `zhou2021-eval/`: re-evaluates those checkpoints, read as an input, without training.
- `zhou2021-ablations/`, `zhou2021-ablations2/`: held-out diagnostics and the paper's Table 1 models (seed 0).

Results are in `results/zhou2021/`.

## Wang 2020

`wang2020/`: builds the patches from the private datasets, then trains CNN A, CNN B, the single views and CNN A from scratch (3 seeds each). Finished runs are never trained again. Results are in `results/wang2020/`.
