import numpy as np


def variance_of_intensities(volume: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Calculate the variance of lesion intensities.

    A high variance indicates a heterogeneous lesion, which is one of the
    echo-pattern signs used to separate malignant from benign masses.

    Args:
        volume (np.ndarray): Intensity array, or the lesion voxels themselves.
        mask (np.ndarray, optional): Binary array selecting the lesion voxels.

    Returns:
        float: Variance of the intensities, or nan if there are none.
    """
    volume = np.asarray(volume)
    values = volume[np.asarray(mask) > 0] if mask is not None else volume.ravel()
    return float(np.var(values)) if values.size else float("nan")
