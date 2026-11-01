"""How far the invariances of the Laplace-Beltrami spectrum survive the voxel pipeline on TDSC-ABUS lesions.

Each lesion is moved by 5 random rigid motions, re-voxelised on an isotropic grid and re-meshed, once with the binary
mask's staircase surface and once after Gaussian smoothing. Motion 0 is repeated on a coarser grid. Each surface gets the
Laplace-Beltrami eigenvalues (L phi = lambda M phi) and the eigenvalues of L alone, as in the ESWA cache.
The Test split's labels are stored but must stay unused while descriptors are being designed.

    python -m abus_classification.lb_invariance
"""
import os
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"   # one process per lesion

import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage
from scipy.sparse.linalg import eigsh
from scipy.spatial.transform import Rotation
from tdsc_abus2023_pytorch import TDSC, DataSplits

from abus_classification.features.shape_descriptor.shape_descriptors import (
    compute_eigendecomposition, compute_laplace_beltrami_operator)
from abus_classification.utils.mesh import get_mesh_from_3d_mask
from abus_classification.utils.spacing import TDSC_SPACING, resample_to_spacing

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "lb-invariance" / "spectra.npz"
TDSC_PATH = os.environ.get("TDSC_PATH", str(ROOT / "data" / "tdsc"))
K, N_MOTIONS, FINE_MM, COARSE_MM = 21, 5, 0.4, 0.8
SMOOTH_MM = 0.8     # Gaussian sigma of the smoothed surfaces, the same on both grids
SURFACES = ("staircase", "smoothed")


def largest_component(mask):
    labels, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    return mask if n < 2 else labels == np.bincount(labels.ravel())[1:].argmax() + 1


def tight(mask):
    pts = np.argwhere(mask)
    return mask[tuple(slice(a, b) for a, b in zip(pts.min(0), pts.max(0) + 1))]


def move(field, rotation, shift):
    """Rigid motion of a float field about its centre, on a grid large enough to hold every rotation."""
    n = int(np.ceil(np.linalg.norm(field.shape))) + 8
    centre_in, centre_out = (np.asarray(field.shape) - 1) / 2, (np.array([n, n, n]) - 1) / 2
    matrix = rotation.as_matrix().T
    return ndimage.affine_transform(field, matrix, offset=centre_in - matrix @ (centre_out + shift),
                                    output_shape=(n, n, n), order=1)


def measure(mask, h, surface):
    """(lambda, eigenvalues of L, area, volume, vertices, Euler number, mesh pieces) of one re-voxelised lesion.

    Each extra piece of a mesh adds a zero eigenvalue and shifts every index, so only the largest piece is kept.
    """
    mask = np.pad(largest_component(mask), 2)
    field = mask.astype(np.float32)
    if surface == "smoothed":
        field = ndimage.gaussian_filter(field, SMOOTH_MM / h)
    mesh = get_mesh_from_3d_mask(field, spacing=(h, h, h), preprocess=False)
    pieces = mesh.split(only_watertight=False)
    mesh = max(pieces, key=lambda piece: piece.area)
    L, M = compute_laplace_beltrami_operator(mesh)
    lam, _ = compute_eigendecomposition(L, M, k=K)
    lam_l = np.sort(eigsh(L, k=K, sigma=-1e-8, which="LM", return_eigenvectors=False))
    return lam, lam_l, mesh.area, abs(mesh.volume), len(mesh.vertices), mesh.euler_number, len(pieces)


def one_lesion(split, index, seed):
    _, mask, label, _ = TDSC(path=TDSC_PATH, split=split, download=False)[index]
    mask = tight(np.asarray(mask) > 0)
    native_vertices = len(get_mesh_from_3d_mask(np.pad(mask, 2).astype(np.uint8), spacing=TDSC_SPACING,
                                                preprocess=False).vertices)
    rng = np.random.default_rng(seed)
    motions = [(Rotation.random(random_state=rng), rng.uniform(-0.5, 0.5, 3)) for _ in range(N_MOTIONS)]
    fine = resample_to_spacing(mask.astype(np.float32), TDSC_SPACING, FINE_MM)[0]
    coarse = resample_to_spacing(mask.astype(np.float32), TDSC_SPACING, COARSE_MM)[0]
    renders = [(fine, FINE_MM, m) for m in motions] + [(coarse, COARSE_MM, motions[0])]
    rows = [[measure(move(f, *m) > 0.5, h, s) for f, h, m in renders] for s in SURFACES]
    get = lambda i: np.array([[r[i] for r in row] for row in rows])   # (surface, render, ...)
    return dict(lam=get(0), lam_l=get(1), area=get(2), volume=get(3), vertices=get(4), euler=get(5), pieces=get(6),
                native_vertices=native_vertices, y=int(label == 0), test=split == DataSplits.TEST)


def main():
    jobs = []
    for split in (DataSplits.TRAIN, DataSplits.VALIDATION, DataSplits.TEST):
        jobs += [(split, i, len(jobs) + i) for i in range(len(TDSC(path=TDSC_PATH, split=split, download=False)))]
    cases = Parallel(n_jobs=-1, verbose=5)(delayed(one_lesion)(*j) for j in jobs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # The ESWA eigenvalues come along so the notebook opens without the 100 MB cache.
    np.savez_compressed(OUT, eswa_lb=np.load(ROOT / "data" / "eswa2025.npz")["lb"],
                        **{k: np.asarray([c[k] for c in cases]) for k in cases[0]})


if __name__ == "__main__":
    main()
