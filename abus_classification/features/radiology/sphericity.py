import numpy as np
from skimage import measure


def sphericity(mask: np.ndarray) -> float:
    """
    Calculate the sphericity of a 3D lesion mask.

    Sphericity is the ratio of the surface area of a sphere with the same
    volume as the lesion to the lesion's own surface area. It is 1.0 for a
    perfect sphere and falls towards 0 as the shape becomes more irregular
    or elongated.

    The mask is zero-padded before meshing. Without the padding a lesion that
    touches the array border produces an open surface, whose area is missing
    the boundary faces, which silently inflates the sphericity.

    Args:
        mask (np.ndarray): A 3D numpy array containing the mask of the lesion.

    Returns:
        float: The sphericity of the lesion, or nan if the mask is empty.
    """
    assert mask.ndim == 3, "The mask should be a 3D ndarray."

    mask = np.asarray(mask) > 0
    volume = float(mask.sum())
    if volume == 0:
        return float("nan")

    verts, faces, _, _ = measure.marching_cubes(np.pad(mask, 1).astype(np.uint8), level=0.5)
    surface_area = measure.mesh_surface_area(verts, faces)
    if surface_area == 0:
        return float("nan")

    return float((np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area)
