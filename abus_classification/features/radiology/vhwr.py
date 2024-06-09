import numpy as np


def height_to_width_ratio(mask: np):
    """
    Calculate the volumetric width-to-height ratio of a 3D lesion mask.

    Args:
        mask (np.ndarray): A 3D numpy array containing the mask of the lesion.

    Returns:
        float: The width-to-height ratio of the lesion.
    """
    # Ensure the mask is a 3D array
    assert mask.ndim == 3, "The mask should be a 3D ndarray."

    # Find the indices of the lesion (where the mask is non-zero)
    lesion_indices = np.argwhere(mask)

    # Get the bounding box of the lesion
    z_min, y_min, x_min = lesion_indices.min(axis=0)
    z_max, y_max, x_max = lesion_indices.max(axis=0)

    # Calculate the dimensions of the bounding box
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    depth = z_max - z_min + 1

    # Calculate the width-to-height ratio
    width_height_ratio = width / height

    return width_height_ratio

