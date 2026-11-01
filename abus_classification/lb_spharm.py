"""Spherical harmonic features of TDSC-ABUS lesions, with the same stability checks as the spectrum (`lb_stability`).

Surfaces with handles are corrected by more smoothing until they are genus 0, and the number of steps is stored.
The Test split's labels are stored but must stay unused while descriptors are being designed.

    python -m abus_classification.lb_spharm
    python -m abus_classification.lb_spharm --predicted
"""
import os
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"   # one process per lesion

import numpy as np
from joblib import Parallel, delayed
from tdsc_abus2023_pytorch import TDSC, DataSplits

from abus_classification.features.shape_descriptor.shape_descriptors import (compute_eigendecomposition,
                                                                              compute_laplace_beltrami_operator)
from abus_classification.features.shape_descriptor.spharm import (map_quality, real_harmonics, spharm_coefficients,
                                                                   spharm_power, sphere_map)
from abus_classification.lb_invariance import TDSC_PATH, largest_component, tight
from abus_classification.lb_roundness import CELLS
from abus_classification.lb_stability import SMOOTH, boundary_noise, on_grid, predicted_masks, surface
from abus_classification.utils.spacing import TDSC_SPACING

OUT = Path(__file__).resolve().parents[1] / "results" / "lb-spharm"
DEGREE, DRAWS, NOISE_MM = 15, 2, (0.25, 0.5, 1.0)
RENDERS = ["clean"] + [f"noise {a} mm #{d}" for a in NOISE_MM for d in range(DRAWS)]


def genus_zero_surface(mask, spacing, side, tries=6):
    """The surface, with smoothing raised by half until it is closed and genus 0. Returns it and the steps taken, or None."""
    for step in range(tries):
        mesh, _ = surface(mask, spacing, side, SMOOTH * 1.5 ** step)
        if mesh.is_watertight and mesh.euler_number == 2:
            return mesh, step
    return None, tries


def measure(mask, spacing, side):
    """Harmonic power, degree-1 axes, reconstruction error, map quality, lambda_1, and Hersch's energy for one mask."""
    mesh, steps = genus_zero_surface(mask, spacing, side)
    if mesh is None:
        return dict(power=np.full(DEGREE + 1, np.nan), axes=np.full(3, np.nan), error=np.full(2, np.nan), quality=np.full(3, np.nan),
                    area=np.nan, volume=np.nan, lam1=np.nan, energy=np.nan, steps=steps)
    sphere = sphere_map(mesh)
    coefficients = spharm_coefficients(mesh, sphere, DEGREE)
    radius = np.sqrt(mesh.area / (4 * np.pi))
    error = np.linalg.norm(real_harmonics(sphere, DEGREE) @ coefficients - mesh.vertices, axis=1) / radius
    L, M = compute_laplace_beltrami_operator(mesh)
    lam = compute_eigendecomposition(L, M, k=3)[0]
    return dict(power=spharm_power(coefficients, DEGREE) / radius ** 2, axes=np.linalg.svd(coefficients[1:4], compute_uv=False) / radius,
                error=np.array([np.median(error), np.percentile(error, 95)]), quality=np.array(map_quality(mesh, sphere)),
                area=mesh.area, volume=abs(mesh.volume), lam1=lam[1], energy=np.einsum("ij,ij->", sphere, L @ sphere), steps=steps)


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
    rows = []
    for mask in [truth] + list(predictions):
        try:
            grid_mask = on_grid(tight(largest_component(mask)), spacing, side, CELLS)[0]
            rows.append(measure(grid_mask, (side / CELLS,) * 3, side))
        except (ValueError, IndexError):    # nothing found, or too small to survive the smoothing: counted as a miss
            rows.append(dict(power=np.full(DEGREE + 1, np.nan), axes=np.full(3, np.nan), error=np.full(2, np.nan),
                             quality=np.full(3, np.nan), area=np.nan, volume=np.nan, lam1=np.nan, energy=np.nan, steps=-1))
    return stack(rows)


def predicted():
    masks, truth, _ = predicted_masks()
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(one_prediction)(truth[i], list(masks[:, i])) for i in range(len(truth)))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "predicted.npz", **{k: np.asarray([r[k] for r in rows]) for k in rows[0]})


if __name__ == "__main__":
    predicted() if "--predicted" in sys.argv else main()
