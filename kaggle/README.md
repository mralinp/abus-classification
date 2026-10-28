# Kaggle kernels for the ESWA 2025 reproduction

Script kernels (GPU + internet). Push one with `kaggle kernels push -p kaggle/<name>`.

- `reproduction/`: clones this branch, finds the TDSC-ABUS splits in the private datasets `simayousef/tdsc-abus2023-trainval` and `-test`, builds the feature cache (about 1.7 h), runs Table 3. Its output holds `eswa2025.npz`.
- `ablations/`, `seeds/`: reuse that cache through `kernel_sources` (no rebuild). Ablations (paper Tables 4-8, seed 0) take about 3 h; seeds 1 and 2 about 2 h.

Results are in `results/eswa2025/`. Edit the kernel `id`s if you push under another Kaggle account.
