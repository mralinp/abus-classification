import argparse

from abus_classification.papers.zhou2021 import main

ap = argparse.ArgumentParser(description="Reproduce Zhou et al. (Med. Image Anal. 2021), CMSVNetIter.")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
a = ap.parse_args()
main(a.epochs, a.seeds).to_csv("zhou2021_results.csv")
