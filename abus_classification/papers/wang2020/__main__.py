import argparse

from abus_classification.papers.wang2020 import main
from abus_classification.papers.wang2020.pipeline import MODES

ap = argparse.ArgumentParser(description="Reproduce Wang et al. (Ultrasound Med. Biol. 2020), multiview Inception-v3.")
ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--out-dir", default=".")
a = ap.parse_args()
main(a.modes, a.seeds, a.out_dir)
