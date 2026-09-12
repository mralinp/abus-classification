import numpy as np


def lesion_volume(mask: np.ndarray, spacing: tuple = None) -> float:
    """
    Calculate the volume of a lesion.

    Args:
        mask (np.ndarray): Binary lesion mask.
        spacing (tuple, optional): Physical voxel size along each axis. When
            omitted every voxel counts as 1, so the result is a voxel count.

    Returns:
        float: Lesion volume in voxels, or in cubed spacing units.
    """
    mask = np.asarray(mask) > 0
    voxel = 1.0 if spacing is None else float(np.prod(spacing))
    if spacing is not None and len(spacing) != mask.ndim:
        raise ValueError(f"spacing has {len(spacing)} entries for a {mask.ndim}D mask")
    return float(mask.sum()) * voxel
