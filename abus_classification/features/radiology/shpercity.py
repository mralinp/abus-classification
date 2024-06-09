import numpy as np
from skimage import measure


def sphericity(mask):
    """
    Calculate the sphericity of a 3D lesion mask.

    Args:
        mask (np.ndarray): A 3D numpy array containing the mask of the lesion.

    Returns:
        float: The sphericity of the lesion.
    """
    # Ensure the mask is a 3D array
    assert mask.ndim == 3, "The mask should be a 3D ndarray."

    # Calculate the volume of the lesion (number of non-zero voxels)
    volume = np.sum(mask)

    # Use marching cubes to calculate the surface area of the lesion
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    surface_area = measure.mesh_surface_area(verts, faces)

    # Calculate the sphericity
    sphericity = (np.pi**(1/3) * (6 * volume)**(2/3)) / surface_area

    return sphericity
