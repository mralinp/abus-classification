import numpy as np


def average_intensity(volume: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Calculate the mean intensity of a lesion.

    Args:
        volume (np.ndarray): Intensity array, or the lesion voxels themselves.
        mask (np.ndarray, optional): Binary array selecting the lesion voxels.

    Returns:
        float: Mean intensity, or nan if there are no voxels to measure.
    """
    volume = np.asarray(volume)
    values = volume[np.asarray(mask) > 0] if mask is not None else volume.ravel()
    return float(np.mean(values)) if values.size else float("nan")
