import argparse

from abus_classification.papers.barekatrezaei2026 import ablations, main

ap = argparse.ArgumentParser(description="Reproduce Barekatrezaei et al. (ESWA 2026), Table 3 or Tables 4-8.")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ablate", action="store_true", help="paper Tables 4-8 instead of Table 3")
a = ap.parse_args()
(ablations if a.ablate else main)(a.epochs, a.seed)
