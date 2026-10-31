import argparse

from abus_classification.papers.yu2024 import main
from abus_classification.papers.yu2024.pipeline import MODES

ap = argparse.ArgumentParser(description="Reproduce Yu et al. (Applied Sciences 2024), ResNet34 with soft and hard voting.")
ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--out-dir", default=".")
ap.add_argument("--paper-schedule", action="store_true", help="fixed 100 epochs, no validation")
a = ap.parse_args()
main(a.modes, a.seeds, a.out_dir, early_stopping=not a.paper_schedule)
