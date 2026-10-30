# Kaggle kernels for the ESWA 2025 reproduction

Script kernels (GPU + internet). Push one with `kaggle kernels push -p kaggle/<name>`.

- `reproduction/`: clones this branch, finds the TDSC-ABUS splits in the private datasets `simayousef/tdsc-abus2023-trainval` and `-test`, builds the feature cache (about 1.7 h), runs Table 3. Its output holds `eswa2025.npz`.
- `ablations/`, `seeds/`: reuse that cache through `kernel_sources` (no rebuild). Ablations (paper Tables 4-8, seed 0) take about 3 h; seeds 1 and 2 about 2 h.

Results are in `results/eswa2025/`. Edit the kernel `id`s if you push under another Kaggle account.

## Zhou 2021

- `zhou2021/`: trains CMSVNetIter (3 seeds x 100 epochs, about 4 h). Its output holds the trained checkpoints.
- `zhou2021-eval/`: re-evaluates those checkpoints, read as an input, without training.
- `zhou2021-ablations/`, `zhou2021-ablations2/`: held-out diagnostics and the paper's Table 1 models (seed 0).

Results are in `results/zhou2021/`.
