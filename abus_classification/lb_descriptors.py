"""Shape-DNA, HKS, WKS and GPS summaries of TDSC-ABUS lesions, with the stability checks of `lb_stability`.

Surfaces are size-matched. The Test split's labels are stored but must stay unused while descriptors are being designed.

    python -m abus_classification.lb_descriptors
    python -m abus_classification.lb_descriptors --predicted
"""
import os
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"   # one process per lesion

import numpy as np
from joblib import Parallel, delayed
from tdsc_abus2023_pytorch import TDSC, DataSplits

from abus_classification.features.shape_descriptor.shape_descriptors import spectral_summaries
from abus_classification.lb_invariance import TDSC_PATH, largest_component, tight
from abus_classification.lb_roundness import CELLS
from abus_classification.lb_stability import boundary_noise, on_grid, predicted_masks, surface
from abus_classification.utils.spacing import TDSC_SPACING

OUT = Path(__file__).resolve().parents[1] / "results" / "lb-descriptors"
DRAWS, NOISE_MM = 2, (0.25, 0.5, 1.0)
RENDERS = ["clean"] + [f"noise {a} mm #{d}" for a in NOISE_MM for d in range(DRAWS)]


def measure(mask, spacing, side):
    mesh, _ = surface(mask, spacing, side)
    return dict(spectral_summaries(mesh), volume=abs(mesh.volume), area=mesh.area, vertices=len(mesh.vertices))


def stack(rows):
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def one_lesion(split, index, seed):
    _, mask, label, _ = TDSC(path=TDSC_PATH, split=split, download=False)[index]
    mask = tight(largest_component(tight(np.asarray(mask) > 0)))
    side = (mask.sum() * np.prod(TDSC_SPACING)) ** (1 / 3)
    clean, grid = on_grid(mask, TDSC_SPACING, side, CELLS)
    rows = [measure(clean, grid, side)]
    for i, amplitude in enumerate(NOISE_MM):
        for draw in range(DRAWS):
            rows.append(measure(boundary_noise(clean, grid, amplitude, seed + 100 * i + draw), grid, side))
    return dict(stack(rows), y=int(label == 0), test=split == DataSplits.TEST)


def main():
    jobs = []
    for split in (DataSplits.TRAIN, DataSplits.VALIDATION, DataSplits.TEST):
        jobs += [(split, i, 1000 * len(jobs) + 1000 * i) for i in range(len(TDSC(path=TDSC_PATH, split=split, download=False)))]
    cases = Parallel(n_jobs=-1, verbose=5)(delayed(one_lesion)(*j) for j in jobs)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "features.npz", renders=np.array(RENDERS), **{k: np.asarray([c[k] for c in cases]) for k in cases[0]})


def one_prediction(truth, predictions, spacing=(0.476, 0.476, 0.476)):
    truth = tight(largest_component(truth))
    side = (truth.sum() * np.prod(spacing)) ** (1 / 3)
    rows, template = [], None
    for mask in [truth] + list(predictions):
        try:
            grid_mask = on_grid(tight(largest_component(mask)), spacing, side, CELLS)[0]
            row = measure(grid_mask, (side / CELLS,) * 3, side)
            template = template or {k: np.full(np.shape(v), np.nan) for k, v in row.items()}
        except (ValueError, IndexError):    # nothing found, or too small to survive the smoothing: counted as a miss
            row = None
        rows.append(row)
    return stack([template if r is None else r for r in rows])


def predicted():
    masks, truth, _ = predicted_masks()
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(one_prediction)(truth[i], list(masks[:, i])) for i in range(len(truth)))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "predicted.npz", **{k: np.asarray([r[k] for r in rows]) for k in rows[0]})


if __name__ == "__main__":
    predicted() if "--predicted" in sys.argv else main()
