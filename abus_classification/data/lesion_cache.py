"""Lesion cache shared by the paper implementations: one row per TDSC-ABUS volume (200: Train, Validation, Test).

Stored in `data/eswa2025.npz` with keys
    split     True for the official Test split
    y         1 = malignant (TDSC label 0), 0 = benign
    radiomic  the six Tan et al. 2012 features (RADIOMIC), millimetre-aware
    lb        first 100 Laplace-Beltrami eigenvalues of the lesion mesh (mm), unnormalised
    vol, mask lesion-centred 128x64x80 patches at 0.476 mm isotropic voxels (float16; volume scaled to [0, 1])
Building it takes about 1.7 h on 4 cores; the Kaggle kernel in `kaggle/reproduction` produced the stored copy.
"""
import os
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage
from scipy.sparse.linalg import eigsh
from tdsc_abus2023_pytorch import TDSC, DataSplits

from abus_classification.features.radiology import extract_radiology_features
from abus_classification.features.shape_descriptor.shape_descriptors import compute_laplace_beltrami_operator
from abus_classification.utils.mesh import lesion_mesh
from abus_classification.utils.spacing import TDSC_SPACING

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "eswa2025.npz"
TDSC_PATH = os.environ.get("TDSC_PATH", str(ROOT / "data" / "tdsc"))  # downloaded here if missing
PATCH, ISO, N_EIG = (128, 64, 80), 0.476, 100
# Tan et al. 2012 six features, computed with the repo's millimetre-aware extractor.
RADIOMIC = ["height_to_width_ratio", "entropy", "mean_intensity", "margin_contrast", "intensity_variance", "sphericity"]


def lb_eigenvalues(mask):
    """First 100 eigenvalues of the cotangent Laplacian L (Eq. 3-4, no mass matrix), mesh in mm."""
    pts = np.argwhere(mask)
    lo, hi = pts.min(0), pts.max(0) + 1
    mesh = lesion_mesh(mask[tuple(slice(a, b) for a, b in zip(lo, hi))], TDSC_SPACING)
    L, _ = compute_laplace_beltrami_operator(mesh)
    return np.sort(eigsh(L, k=N_EIG, sigma=-1e-8, which="LM", return_eigenvectors=False))


def patch(vol, mask):
    """128x64x80 VOI and mask at 0.476 mm isotropic voxels, centred on the lesion."""
    centre = np.argwhere(mask).mean(0)
    scale = ISO / np.asarray(TDSC_SPACING)                 # source voxels per output voxel
    half = np.asarray(PATCH) * scale / 2
    lo = np.maximum(np.floor(centre - half).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(centre + half).astype(int) + 3, vol.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    offset = centre - half + 0.5 * scale - 0.5 - lo
    out = []
    for a in (vol[box], mask[box]):                        # anti-alias, then spline-resample
        a = ndimage.gaussian_filter(a.astype(np.float32), 0.5 * scale)
        out.append(ndimage.affine_transform(a, scale, offset=offset, output_shape=PATCH, order=1))
    return (out[0] / 255).astype(np.float16), (out[1] > 0.5).astype(np.float16)


def one_case(split, i):
    vol, mask, label, _ = TDSC(path=TDSC_PATH, split=split, download=False)[i]
    v, m = patch(vol, mask)
    feats = extract_radiology_features(vol, mask)           # once: it computes every feature
    return dict(split=split.name == "TEST", y=int(label == 0),   # y: 1 = malignant
                radiomic=[feats[k] for k in RADIOMIC], lb=lb_eigenvalues(mask), vol=v, mask=m)


def build_cache():
    CACHE.parent.mkdir(parents=True, exist_ok=True)         # fail now, not after hours of work
    jobs = []
    for split in (DataSplits.TRAIN, DataSplits.VALIDATION, DataSplits.TEST):
        ds = TDSC(path=TDSC_PATH, split=split, download=not (Path(TDSC_PATH) / str(split)).exists())
        jobs += [(split, i) for i in range(len(ds))]
    cases = Parallel(n_jobs=-1, verbose=5)(delayed(one_case)(*j) for j in jobs)   # input order is kept
    np.savez_compressed(CACHE, **{k: np.asarray([c[k] for c in cases]) for k in cases[0]})


def load_raw():
    """The cache as a dict of arrays (built first if missing) and the Train+Validation / Test index arrays."""
    if not CACHE.exists():
        build_cache()
    d = dict(np.load(CACHE))
    test = d["split"]
    return d, np.flatnonzero(~test), np.flatnonzero(test)
