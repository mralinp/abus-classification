"""Roundness and lobulation of TDSC-ABUS lesions from size-matched surfaces.

Every lesion gets its own grid and smoothing, proportional to the cube root of its volume, so all meshes have about the
same vertex count and the same relative detail. This removes the size dependence that a fixed voxel size and smoothing
width put into lambda * area. Small lesions are still limited by the 0.476 mm slice spacing of the scans.
The Test split's labels are stored but must stay unused while descriptors are being designed.

    python -m abus_classification.lb_roundness
    python -m abus_classification.lb_roundness --samples   # after the above: meshes for the 3D figures
"""
import os
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"   # one process per lesion

import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage
from tdsc_abus2023_pytorch import TDSC, DataSplits

from abus_classification.features.shape_descriptor.shape_descriptors import lb_spectrum
from abus_classification.lb_invariance import TDSC_PATH, largest_component, tight
from abus_classification.utils.mesh import get_mesh_from_3d_mask
from abus_classification.utils.spacing import TDSC_SPACING, resample_to_spacing

OUT = Path(__file__).resolve().parents[1] / "results" / "lb-roundness" / "spectra.npz"
K, CELLS = 200, 24                     # grid step = (volume)^(1/3) / CELLS
SMOOTHING = (0.04, 0.08)               # Gaussian sigma, as a fraction of (volume)^(1/3)


def size_matched_mesh(mask, sigma_fraction):
    """Largest mesh piece of a lesion on its own grid, and the number of pieces, in mm."""
    mask = tight(largest_component(tight(np.asarray(mask) > 0)))
    side = (mask.sum() * np.prod(TDSC_SPACING)) ** (1 / 3)
    field, spacing = resample_to_spacing(mask.astype(np.float32), TDSC_SPACING, side / CELLS)
    field = ndimage.gaussian_filter(np.pad(field, CELLS // 2), sigma_fraction * CELLS)
    pieces = get_mesh_from_3d_mask(field, spacing=spacing, preprocess=False).split(only_watertight=False)
    return max(pieces, key=lambda piece: piece.area), len(pieces)


def measure(mask, sigma_fraction):
    mesh, pieces = size_matched_mesh(mask, sigma_fraction)
    lam = lb_spectrum(mesh, min(K, len(mesh.vertices) - 2), scale_invariant=False)
    return (np.pad(lam, (0, K - len(lam)), constant_values=np.nan), mesh.area, abs(mesh.volume), len(mesh.vertices),
            mesh.euler_number, pieces, np.ptp(mesh.principal_inertia_vectors @ mesh.vertices.T, axis=1))


def one_lesion(split, index):
    _, mask, label, _ = TDSC(path=TDSC_PATH, split=split, download=False)[index]
    rows = [measure(mask, s) for s in SMOOTHING]
    get = lambda i: np.array([r[i] for r in rows])    # (smoothing, ...)
    return dict(lam=get(0), area=get(1), volume=get(2), vertices=get(3), euler=get(4), pieces=get(5), extents=get(6),
                y=int(label == 0), test=split == DataSplits.TEST)


def main():
    jobs = []
    for split in (DataSplits.TRAIN, DataSplits.VALIDATION, DataSplits.TEST):
        jobs += [(split, i) for i in range(len(TDSC(path=TDSC_PATH, split=split, download=False)))]
    cases = Parallel(n_jobs=-1, verbose=5)(delayed(one_lesion)(*j) for j in jobs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, smoothing=np.array(SMOOTHING), **{k: np.asarray([c[k] for c in cases]) for k in cases[0]})


def save_samples():
    """Size-matched meshes of Train lesions at 5 percentiles of lambda_1 * area, for each class, picked by that rule."""
    data = np.load(OUT)
    n_train = len(TDSC(path=TDSC_PATH, split=DataSplits.TRAIN, download=False))
    ratio, y = (data["lam"][:, 0, 1] * data["area"][:, 0] / (8 * np.pi))[:n_train], data["y"][:n_train]
    genus0 = data["euler"][:n_train, 0] == 2
    out = {}
    for label, name in ((1, "malignant"), (0, "benign")):
        members = np.flatnonzero((y == label) & genus0)
        for q in (5, 25, 50, 75, 95):
            index = int(members[np.abs(ratio[members] - np.percentile(ratio[members], q)).argmin()])
            _, mask, _, _ = TDSC(path=TDSC_PATH, split=DataSplits.TRAIN, download=False)[index]
            mesh, _ = size_matched_mesh(mask, SMOOTHING[0])
            key = f"{name}_q{q}"
            out[key + "_vertices"], out[key + "_faces"] = mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)
            out[key + "_case"], out[key + "_ratio"] = index, ratio[index]
    np.savez_compressed(OUT.parent / "sample_meshes.npz", **out)


if __name__ == "__main__":
    save_samples() if "--samples" in sys.argv else main()
