import argparse

from abus_classification.papers.hejduk2022 import main

ap = argparse.ArgumentParser(description="Reproduce Hejduk et al. (Eur. Radiol. 2022), step-wise sliding-window dCNNs.")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--lr", type=float, default=1e-6)
ap.add_argument("--stride", type=int, default=10)
ap.add_argument("--slice-step", type=int, default=1)
ap.add_argument("--out-dir", default=".")
ap.add_argument("--paper-schedule", action="store_true", help="fixed epochs, no validation")
a = ap.parse_args()
main(a.seeds, a.out_dir, a.lr, a.stride, a.slice_step, not a.paper_schedule)
