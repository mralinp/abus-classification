import numpy as np
from skimage import measure


def compactness(mask: np.ndarray) -> float:
    """
    Calculate the compactness of a lesion mask.

    Compactness compares the lesion's surface area (3D) or perimeter (2D)
    against that of the most compact shape enclosing the same volume or area.
    It is 1.0 for a perfect sphere or circle and grows without bound as the
    boundary becomes more convoluted, so it reads as the inverse of
    :func:`~abus_classification.features.radiology.sphericity`.

    Args:
        mask (np.ndarray): A 2D or 3D binary array containing the lesion mask.

    Returns:
        float: The compactness of the lesion, or nan if the mask is empty.
    """
    mask = np.asarray(mask) > 0
    if not mask.any():
        return float("nan")

    if mask.ndim == 2:
        area = float(mask.sum())
        perimeter = measure.perimeter(mask)
        if area == 0:
            return float("nan")
        return float(perimeter ** 2 / (4 * np.pi * area))

    if mask.ndim == 3:
        volume = float(mask.sum())
        verts, faces, _, _ = measure.marching_cubes(np.pad(mask, 1).astype(np.uint8), level=0.5)
        surface_area = measure.mesh_surface_area(verts, faces)
        if volume == 0:
            return float("nan")
        return float(surface_area ** 3 / (36 * np.pi * volume ** 2))

    raise ValueError(f"mask must be 2D or 3D, got {mask.ndim}D")
