import numpy as np
from skimage import measure


def compactness(mask: np.ndarray, spacing=None) -> float:
    """
    Calculate the compactness of a lesion mask.

    Compactness compares the lesion's surface area (3D) or perimeter (2D)
    against that of the most compact shape enclosing the same volume or area.
    It is 1.0 for a perfect sphere or circle and grows without bound as the
    boundary becomes more convoluted, so it reads as the inverse of
    :func:`~abus_classification.features.radiology.sphericity`.

    Marching cubes overestimates surface area on a voxel staircase, and more
    so when voxels are anisotropic: on TDSC-ABUS spacing a sphere scores
    1.83 even with `spacing`, against 1.30 after resampling to 0.6 mm
    cubes. Resample to isotropic voxels first for values comparable across
    datasets (``utils.spacing.prepare_lesion`` does this, and
    ``extract_radiology_features`` uses it).

    Args:
        mask (np.ndarray): A 2D or 3D binary array containing the lesion mask.
        spacing (tuple, optional): Physical voxel size along each axis. Needed
            for anisotropic 3D data; 2D masks must have square pixels.

    Returns:
        float: The compactness of the lesion, or nan if the mask is empty.
    """
    mask = np.asarray(mask) > 0
    if not mask.any():
        return float("nan")

    if mask.ndim == 2:
        if spacing is not None and not np.isclose(spacing[0], spacing[1]):
            raise ValueError("2D compactness needs square pixels; resample first")
        area = float(mask.sum())
        perimeter = measure.perimeter(mask)
        return float(perimeter ** 2 / (4 * np.pi * area))

    if mask.ndim == 3:
        spacing = (1.0, 1.0, 1.0) if spacing is None else tuple(float(s) for s in spacing)
        volume = float(mask.sum()) * float(np.prod(spacing))
        verts, faces, _, _ = measure.marching_cubes(np.pad(mask, 1).astype(np.uint8), level=0.5, spacing=spacing)
        surface_area = measure.mesh_surface_area(verts, faces)
        return float(surface_area ** 3 / (36 * np.pi * volume ** 2))

    raise ValueError(f"mask must be 2D or 3D, got {mask.ndim}D")
