"""How far the Laplace-Beltrami spectrum of a TDSC-ABUS lesion moves under the errors real data has.

Every lesion goes through the size-matched surface of `lb_roundness`, then gets simulated boundary noise, a coarser and a
finer grid, and thicker scan slices. Test lesions also get a real segmentation error: Zhou 2021's predicted masks, with its
stored weights read only, on patches centred on the ground truth. Only geometry is used there, not Test labels.

    python -m abus_classification.lb_stability
    python -m abus_classification.lb_stability --predicted
    python -m abus_classification.lb_stability --samples     # after both: meshes for the 3D figures
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
from abus_classification.lb_roundness import CELLS
from abus_classification.utils.mesh import get_mesh_from_3d_mask
from abus_classification.utils.spacing import TDSC_SPACING, resample_to_spacing

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "lb-stability"
K, SMOOTH = 101, 0.04                     # as `lb_roundness`: sigma is a fraction of (volume)^(1/3)
NOISE_MM, NOISE_LENGTH_MM, DRAWS = (0.25, 0.5, 1.0), 1.5, 3
GRIDS, SLICE_FACTORS = (16, 32), (2, 4)
RENDERS = ["clean"] + [f"noise {a} mm #{d}" for a in NOISE_MM for d in range(DRAWS)] \
    + [f"grid {g}" for g in GRIDS] + [f"slices x{f}" for f in SLICE_FACTORS]


def thicken(mask, factor):
    """The mask as a scan with `factor` times the slice spacing would see it, on the original grid."""
    block = np.repeat(np.add.reduceat(mask.astype(float), np.arange(0, mask.shape[2], factor), axis=2), factor, axis=2)
    counts = np.repeat(np.diff(np.r_[np.arange(0, mask.shape[2], factor), mask.shape[2]]), factor)
    return (block[:, :, :mask.shape[2]] / counts[:mask.shape[2]]) >= 0.5


def boundary_noise(mask, spacing, amplitude_mm, seed):
    """Moves the boundary by smooth noise with this standard deviation: a threshold of signed distance plus noise."""
    distance = ndimage.distance_transform_edt(~mask, spacing) - ndimage.distance_transform_edt(mask, spacing)
    noise = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=mask.shape), NOISE_LENGTH_MM / np.asarray(spacing))
    return distance + amplitude_mm * noise / noise.std() < 0


def surface(mask, spacing, side, smooth=SMOOTH):
    """Largest mesh piece of the smoothed mask's largest component, and the number of pieces."""
    field = ndimage.gaussian_filter(np.pad(largest_component(mask).astype(np.float32), 8), smooth * side / np.mean(spacing))
    pieces = get_mesh_from_3d_mask(field, spacing=spacing, preprocess=False).split(only_watertight=False)
    return max(pieces, key=lambda piece: piece.area), len(pieces)


def measure(mask, spacing, side, reference=None):
    """Spectrum and geometry of a binary mask on an isotropic grid. Dice is against `reference` on that grid."""
    dice = 2 * (mask & reference).sum() / max(mask.sum() + reference.sum(), 1) if reference is not None else np.nan
    mesh, pieces = surface(mask, spacing, side)
    lam = lb_spectrum(mesh, min(K, len(mesh.vertices) - 2), scale_invariant=False)
    return (np.pad(lam, (0, K - len(lam)), constant_values=np.nan), mesh.area, abs(mesh.volume), len(mesh.vertices),
            mesh.euler_number, pieces, np.ptp(mesh.principal_inertia_vectors @ mesh.vertices.T, axis=1), dice)


def on_grid(mask, spacing, side, cells):
    field, actual = resample_to_spacing(mask.astype(np.float32), spacing, side / cells)
    return np.pad(field > 0.5, cells // 2), actual


def renders(mask, spacing, seed):
    """Every perturbed version of one lesion, in the order of RENDERS."""
    side = (mask.sum() * np.prod(spacing)) ** (1 / 3)
    clean, grid = on_grid(mask, spacing, side, CELLS)
    rows = [measure(clean, grid, side)]
    for i, amplitude in enumerate(NOISE_MM):
        for draw in range(DRAWS):
            rows.append(measure(boundary_noise(clean, grid, amplitude, seed + 100 * i + draw), grid, side, clean))
    for cells in GRIDS:
        rows.append(measure(*on_grid(mask, spacing, side, cells), side))
    for factor in SLICE_FACTORS:
        thick, thick_grid = on_grid(thicken(mask, factor), spacing, side, CELLS)
        rows.append(measure(thick, thick_grid, side, clean if thick.shape == clean.shape else None))
    return rows


def stack(rows):
    names = ("lam", "area", "volume", "vertices", "euler", "pieces", "extents", "dice")
    return {n: np.array([r[i] for r in rows]) for i, n in enumerate(names)}


def one_lesion(split, index, seed):
    _, mask, label, _ = TDSC(path=TDSC_PATH, split=split, download=False)[index]
    mask = tight(largest_component(tight(np.asarray(mask) > 0)))
    return dict(stack(renders(mask, TDSC_SPACING, seed)), y=int(label == 0), test=split == DataSplits.TEST)


def main():
    jobs = []
    for split in (DataSplits.TRAIN, DataSplits.VALIDATION, DataSplits.TEST):
        jobs += [(split, i, 1000 * len(jobs) + 1000 * i) for i in range(len(TDSC(path=TDSC_PATH, split=split, download=False)))]
    cases = Parallel(n_jobs=-1, verbose=5)(delayed(one_lesion)(*j) for j in jobs)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "spectra.npz", renders=np.array(RENDERS), **{k: np.asarray([c[k] for c in cases]) for k in cases[0]})


def predicted_masks(seeds=(0, 1, 2), batch=2):
    """Zhou 2021 masks for the Test patches, (seeds, 70, patch), and the ground truth. Weights are only read."""
    import torch
    from abus_classification.data.lesion_cache import load_raw
    from abus_classification.papers.zhou2021.model import MultiTaskVNet

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data, _, test = load_raw()
    masks = []
    for seed in seeds:
        saved = torch.load(ROOT / "data" / "zhou2021_checkpoints" / f"zhou2021_seed{seed}.pt", map_location="cpu", weights_only=False)
        net = MultiTaskVNet().to(device)
        net.load_state_dict(saved.get("net", saved) if isinstance(saved, dict) else saved)
        net.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(test), batch):
                v = torch.as_tensor(np.asarray(data["vol"][test[start:start + batch]], np.float32), device=device)
                prob = torch.zeros_like(v)
                for _ in range(2):                  # the paper's two refinement passes
                    prob = torch.sigmoid(net(v[:, None] + prob[:, None])[0])
                out.append(prob.cpu().numpy() > 0.5)
        masks.append(np.concatenate(out))
    return np.array(masks), np.asarray(data["mask"][test]) > 0.5, data["vol"].shape[1:]


def one_prediction(truth, predictions, spacing=(0.476, 0.476, 0.476)):
    """Clean surface of the true patch mask, then of each prediction on the same grid."""
    truth = tight(largest_component(truth))
    side = (truth.sum() * np.prod(spacing)) ** (1 / 3)
    rows = []
    missing = (np.full(K, np.nan), np.nan, np.nan, 0, 0, 0, np.full(3, np.nan), 0.0)
    for mask in [truth] + list(predictions):
        try:
            rows.append(measure(on_grid(tight(largest_component(mask)), spacing, side, CELLS)[0], (side / CELLS,) * 3, side))
        except (ValueError, IndexError):    # nothing found, or too small to survive the smoothing: counted as a miss
            rows.append(missing)
    return rows


def predicted():
    masks, truth, _ = predicted_masks()
    dice = (2 * (masks & truth[None]).sum((2, 3, 4)) / np.maximum(masks.sum((2, 3, 4)) + truth.sum((1, 2, 3)), 1))
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(one_prediction)(truth[i], list(masks[:, i])) for i in range(len(truth)))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "predicted.npz", dice=dice.T, **{k: np.asarray([stack(r)[k] for r in rows]) for k in stack(rows[0]) if k != "dice"})


def save_samples():
    """Meshes for the 3D figures: median-volume Train lesions with noise, and Test lesions at 3 Dice levels of seed 0."""
    out = {}
    def keep(key, mesh):
        out[key + "_vertices"], out[key + "_faces"] = mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)

    data = np.load(OUT / "spectra.npz")
    n_train = len(TDSC(path=TDSC_PATH, split=DataSplits.TRAIN, download=False))
    for label, name in ((1, "malignant"), (0, "benign")):
        members = np.flatnonzero(data["y"][:n_train] == label)
        volume = data["volume"][members, 0]
        index = int(members[np.abs(volume - np.median(volume)).argmin()])
        _, mask, _, _ = TDSC(path=TDSC_PATH, split=DataSplits.TRAIN, download=False)[index]
        mask = tight(largest_component(tight(np.asarray(mask) > 0)))
        side = (mask.sum() * np.prod(TDSC_SPACING)) ** (1 / 3)
        clean, grid = on_grid(mask, TDSC_SPACING, side, CELLS)
        keep(f"{name}_clean", surface(clean, grid, side)[0])
        for amplitude in NOISE_MM:
            keep(f"{name}_noise{amplitude}", surface(boundary_noise(clean, grid, amplitude, index), grid, side)[0])
    masks, truth, _ = predicted_masks(seeds=(0,))
    dice = 2 * (masks[0] & truth).sum((1, 2, 3)) / np.maximum(masks[0].sum((1, 2, 3)) + truth.sum((1, 2, 3)), 1)
    order = np.argsort(dice)
    for name, i in (("low", order[len(order) // 10]), ("middle", order[len(order) // 2]), ("high", order[-len(order) // 10])):
        for kind, mask in (("truth", truth[i]), ("predicted", masks[0, i])):
            mask = tight(largest_component(mask))
            side = (truth[i].sum() * 0.476 ** 3) ** (1 / 3)
            grid_mask, grid = on_grid(mask, (0.476,) * 3, side, CELLS)
            keep(f"{name}_{kind}", surface(grid_mask, (side / CELLS,) * 3, side)[0])
        out[f"{name}_dice"] = dice[i]
    np.savez_compressed(OUT / "sample_meshes.npz", **out)


if __name__ == "__main__":
    if "--samples" in sys.argv:
        save_samples()
    elif "--predicted" in sys.argv:
        predicted()
    else:
        main()
