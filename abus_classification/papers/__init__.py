"""One subpackage per reproduced paper (run with `python -m abus_classification.papers.<name>`).

Shared pieces live elsewhere: `abus_classification.data` (lesion cache), `abus_classification.protocol` (split,
metrics, folds), `features`, `models`, `transformers`, `preprocessing`, `utils`.

    tan2012, tan2013   `abus_classification.features.tan`
    barekatrezaei2026  Laplace-Beltrami spectra + dual-path CNN + stacking
    zhou2021           iterative feature-refining multi-task V-Net
    wang2020           multiview Inception-v3 on coronal and transverse patches
    yu2024             2D ResNet34 on tumor slices with hard and soft voting
    hejduk2022         step-wise dCNNs and a sliding window over slices
"""
