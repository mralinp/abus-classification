import numpy as np
from skimage import measure


def sphericity(mask: np.ndarray, spacing=None) -> float:
    """
    Calculate the sphericity of a 3D lesion mask.

    Sphericity is the ratio of the surface area of a sphere with the same
    volume as the lesion to the lesion's own surface area. It is 1.0 for a
    perfect sphere and falls towards 0 as the shape becomes more irregular
    or elongated.

    Pass `spacing` for anisotropic voxels. TDSC-ABUS voxels measure
    0.2 x 0.073 x 0.476 mm, so without it a sphere is measured as a flattened
    ellipsoid and scores far below 1.

    Marching cubes overestimates surface area on a voxel staircase, and more
    so when voxels are anisotropic: on TDSC-ABUS spacing a sphere scores
    0.82 even with `spacing`, against 0.92 after resampling to 0.6 mm
    cubes. Resample to isotropic voxels first for values comparable across
    datasets (``utils.spacing.prepare_lesion`` does this, and
    ``extract_radiology_features`` uses it).

    The mask is zero-padded before meshing. Without the padding a lesion that
    touches the array border produces an open surface, whose area is missing
    the boundary faces, which silently inflates the sphericity.

    Args:
        mask (np.ndarray): A 3D numpy array containing the mask of the lesion.
        spacing (tuple, optional): Physical voxel size along each axis.
            Defaults to isotropic unit voxels.

    Returns:
        float: The sphericity of the lesion, or nan if the mask is empty.
    """
    mask = np.asarray(mask) > 0
    assert mask.ndim == 3, "The mask should be a 3D ndarray."
    spacing = (1.0, 1.0, 1.0) if spacing is None else tuple(float(s) for s in spacing)

    volume = float(mask.sum()) * float(np.prod(spacing))
    if volume == 0:
        return float("nan")

    verts, faces, _, _ = measure.marching_cubes(np.pad(mask, 1).astype(np.uint8), level=0.5, spacing=spacing)
    surface_area = measure.mesh_surface_area(verts, faces)
    if surface_area == 0:
        return float("nan")

    return float((np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area)
