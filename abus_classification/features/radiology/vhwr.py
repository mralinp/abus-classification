import numpy as np


def height_to_width_ratio(mask: np.ndarray) -> float:
    """
    Calculate the volumetric height-to-width ratio of a 3D lesion mask.

    Arrays follow the (z, y, x) ordering used throughout this project, so
    height is the extent along y (axis 1) and width the extent along x
    (axis 2). A ratio above 1 means the lesion is taller than it is wide,
    the "taller-than-wide" sign associated with malignancy in breast
    ultrasound.

    Args:
        mask (np.ndarray): A 3D numpy array containing the mask of the lesion.

    Returns:
        float: height / width, or nan if the mask is empty.
    """
    assert mask.ndim == 3, "The mask should be a 3D ndarray."

    lesion_indices = np.argwhere(np.asarray(mask) > 0)
    if lesion_indices.size == 0:
        return float("nan")

    _, height, width = np.ptp(lesion_indices, axis=0) + 1
    return float(height) / float(width)
