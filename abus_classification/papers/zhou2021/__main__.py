import argparse

from abus_classification.papers.zhou2021 import main

ap = argparse.ArgumentParser(description="Reproduce Zhou et al. (Med. Image Anal. 2021), CMSVNetIter.")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--ckpt-dir", default=".", help="checkpoints (resume if present) and per-seed probabilities")
ap.add_argument("--val", action="store_true", help="hold out 20%% of Train+Validation and log val AUC per epoch")
a = ap.parse_args()
main(a.epochs, a.seeds, a.ckpt_dir, a.val).to_csv("zhou2021_results.csv")
