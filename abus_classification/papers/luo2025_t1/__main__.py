import argparse

from abus_classification.papers.luo2025_t1 import main

ap = argparse.ArgumentParser(description="Reproduce TDSC-ABUS challenge team T1 (SZU): detection, segmentation, classification.")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--out-dir", default=".")
ap.add_argument("--max-minutes", type=float, default=90, help="time limit per training")
a = ap.parse_args()
main(a.seeds, a.out_dir, max_seconds=a.max_minutes * 60)
