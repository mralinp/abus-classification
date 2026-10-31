"""What the Laplace-Beltrami spectrum of a TDSC-ABUS lesion determines: inputs for Weyl's law and the heat trace.

Each lesion is resampled to 0.4 mm and meshed three ways (a staircase surface and two Gaussian smoothing levels). Only
the largest mesh piece is kept. Each surface gets up to 300 eigenvalues, plus the geometry the spectrum should recover.
The Test split's labels are stored but must stay unused while descriptors are being designed.

    python -m abus_classification.lb_spectrum_content
    python -m abus_classification.lb_spectrum_content --samples   # after the above: meshes for the 3D figures
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

OUT = Path(__file__).resolve().parents[1] / "results" / "lb-spectrum-content" / "spectra.npz"
K, H_MM = 300, 0.4
SMOOTHING_MM = (0.0, 0.4, 0.8)   # 0 is the staircase surface


def measure(mask, sigma_mm):
    """(eigenvalues padded with NaN to K, area, volume, vertices, Euler number, int K^2 dA, mesh pieces)."""
    field = np.pad(mask, 2).astype(np.float32)
    if sigma_mm:
        field = ndimage.gaussian_filter(field, sigma_mm / H_MM)
    pieces = get_mesh_from_3d_mask(field, spacing=(H_MM,) * 3, preprocess=False).split(only_watertight=False)
    mesh = max(pieces, key=lambda piece: piece.area)
    lam = lb_spectrum(mesh, min(K, len(mesh.vertices) - 2), scale_invariant=False)
    mass = np.bincount(mesh.faces.ravel(), np.repeat(mesh.area_faces / 3, 3), minlength=len(mesh.vertices))
    # Gaussian curvature at a vertex is its angle defect per unit of its area.
    curvature_sq = (mesh.vertex_defects ** 2 / mass).sum()
    return (np.pad(lam, (0, K - len(lam)), constant_values=np.nan), mesh.area, abs(mesh.volume), len(mesh.vertices),
            mesh.euler_number, curvature_sq, len(pieces))


def one_lesion(split, index):
    _, mask, label, _ = TDSC(path=TDSC_PATH, split=split, download=False)[index]
    fine = resample_to_spacing(tight(np.asarray(mask) > 0).astype(np.float32), TDSC_SPACING, H_MM)[0]
    fine = largest_component(fine > 0.5)
    rows = [measure(fine, sigma) for sigma in SMOOTHING_MM]
    get = lambda i: np.array([r[i] for r in rows])    # (surface, ...)
    return dict(lam=get(0), area=get(1), volume=get(2), vertices=get(3), euler=get(4), curvature_sq=get(5),
                pieces=get(6), y=int(label == 0), test=split == DataSplits.TEST)


def main():
    jobs = []
    for split in (DataSplits.TRAIN, DataSplits.VALIDATION, DataSplits.TEST):
        jobs += [(split, i) for i in range(len(TDSC(path=TDSC_PATH, split=split, download=False)))]
    cases = Parallel(n_jobs=-1, verbose=5)(delayed(one_lesion)(*j) for j in jobs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, smoothing_mm=np.array(SMOOTHING_MM),
                        **{k: np.asarray([c[k] for c in cases]) for k in cases[0]})


def save_samples():
    """Meshes of 6 Train lesions at the 25th, 50th and 75th volume percentiles of each class, picked without looking."""
    data = np.load(OUT)
    train = DataSplits.TRAIN
    n_train = len(TDSC(path=TDSC_PATH, split=train, download=False))
    volume, y = data["volume"][:n_train, 2], data["y"][:n_train]
    out = {}
    for label, name in ((1, "malignant"), (0, "benign")):
        members = np.flatnonzero(y == label)
        for q in (25, 50, 75):
            index = int(members[np.abs(volume[members] - np.percentile(volume[members], q)).argmin()])
            _, mask, _, _ = TDSC(path=TDSC_PATH, split=train, download=False)[index]
            fine = resample_to_spacing(tight(np.asarray(mask) > 0).astype(np.float32), TDSC_SPACING, H_MM)[0]
            fine = np.pad(largest_component(fine > 0.5), 2).astype(np.float32)
            for surface, field in (("staircase", fine), ("smoothed", ndimage.gaussian_filter(fine, 0.4 / H_MM))):
                pieces = get_mesh_from_3d_mask(field, spacing=(H_MM,) * 3, preprocess=False).split(only_watertight=False)
                mesh = max(pieces, key=lambda piece: piece.area)
                key = f"{name}_q{q}_{surface}"
                out[key + "_vertices"], out[key + "_faces"] = mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)
            out[f"{name}_q{q}_case"] = index
    np.savez_compressed(OUT.parent / "sample_meshes.npz", **out)


if __name__ == "__main__":
    save_samples() if "--samples" in sys.argv else main()
