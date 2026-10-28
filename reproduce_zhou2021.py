"""Back-compat shim (Kaggle kernel): the code now lives in abus_classification.papers.zhou2021.

    uv run python -m abus_classification.papers.zhou2021 [--epochs N] [--seeds 0 1 2]
"""
from abus_classification.papers.zhou2021 import main, train_eval  # noqa: F401

if __name__ == "__main__":
    import runpy
    runpy.run_module("abus_classification.papers.zhou2021", run_name="__main__")
