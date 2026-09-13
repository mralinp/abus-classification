import numpy as np
from skimage import measure

from abus_classification.features.radiology.volume import lesion_volume


def surface_area(mask: np.ndarray, spacing: tuple = None) -> float:
    """
    Calculate the surface area of a 3D lesion from its marching-cubes mesh.

    The mask is zero-padded by one voxel first, so lesions touching the array
    border still produce a closed surface. Without it the boundary faces are
    missing and the area is underestimated.

    Marching cubes overestimates surface area on a voxel staircase, and more
    so when voxels are anisotropic: on TDSC-ABUS spacing a sphere gets
    an area ~12% too large even with `spacing`, and the correct area after resampling to 0.6 mm
    cubes. Resample to isotropic voxels first for values comparable across
    datasets (``utils.spacing.prepare_lesion`` does this, and
    ``extract_radiology_features`` uses it).

    Args:
        mask (np.ndarray): A 3D binary lesion mask.
        spacing (tuple, optional): Physical voxel size along each axis.

    Returns:
        float: Surface area in voxel faces or squared spacing units, or nan
        if the mask is empty.
    """
    mask = np.asarray(mask) > 0
    assert mask.ndim == 3, "The mask should be a 3D ndarray."
    if not mask.any():
        return float("nan")

    verts, faces, _, _ = measure.marching_cubes(
        np.pad(mask, 1).astype(np.uint8), level=0.5,
        spacing=(1.0, 1.0, 1.0) if spacing is None else tuple(spacing),
    )
    return float(measure.mesh_surface_area(verts, faces))


def surface_to_volume_ratio(mask: np.ndarray, spacing: tuple = None) -> float:
    """
    Calculate the lesion's surface area per unit volume.

    For a given volume the most compact shape has the smallest surface, so a
    higher ratio means a more irregular, lobulated or spiculated boundary.
    The ratio is scale-dependent: small lesions score higher than large ones
    of the same shape, which is why sphericity is the size-free counterpart.

    Args:
        mask (np.ndarray): A 3D binary lesion mask.
        spacing (tuple, optional): Physical voxel size along each axis.

    Returns:
        float: Surface area divided by volume, or nan if the mask is empty.
    """
    volume = lesion_volume(mask, spacing)
    if volume == 0:
        return float("nan")
    return surface_area(mask, spacing) / volume
