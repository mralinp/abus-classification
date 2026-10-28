"""Back-compat shim (Kaggle kernels, notebook 2): the code now lives in abus_classification.papers.barekatrezaei2026.

    uv run python -m abus_classification.papers.barekatrezaei2026 [--epochs N] [--seed S] [--ablate]
"""
from abus_classification.data.lesion_cache import CACHE, build_cache  # noqa: F401
from abus_classification.papers.barekatrezaei2026 import ablations, load, main  # noqa: F401
from abus_classification.protocol import DEVICE, metrics  # noqa: F401

if __name__ == "__main__":
    import runpy
    runpy.run_module("abus_classification.papers.barekatrezaei2026", run_name="__main__")
