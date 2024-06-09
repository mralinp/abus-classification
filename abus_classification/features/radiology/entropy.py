import numpy as np
from scipy import stats


def entropy(lesion):
    """
    Calculate the entropy of a 3D lesion mask.

    Args:
        mask (np.ndarray): A 3D numpy array containing the mask of the lesion.

    Returns:
        float: The entropy of the lesion mask.
    """
    # Ensure the mask is a 3D array
    assert lesion.ndim == 3, "The lesion should be a 3D ndarray."

    # Flatten the 3D mask to a 1D array
    mask_flat = lesion.flatten()

    # Calculate the histogram of the voxel intensities
    hist, bin_edges = np.histogram(mask_flat, bins=256, range=(0, 256), density=True)

    # Normalize the histogram to get the probability distribution
    p = hist[hist > 0]  # We only consider non-zero probabilities

    # Calculate the entropy
    ent = stats.entropy(p)

    return ent
