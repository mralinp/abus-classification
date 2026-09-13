import numpy as np
from scipy import stats


def entropy(volume: np.ndarray, mask: np.ndarray = None, bins: int = 256) -> float:
    """
    Calculate the Shannon entropy of the intensity distribution of a lesion.

    Entropy measures how disordered the echo pattern is: a uniform region has
    low entropy, a heterogeneous one high entropy.

    Pass `mask` to restrict the histogram to lesion voxels. Without it every
    voxel of `volume` is counted, so any background included in the array
    contributes its own mode and depresses the result.

    Args:
        volume (np.ndarray): Intensity array of any dimensionality.
        mask (np.ndarray, optional): Binary array selecting the lesion voxels.
        bins (int): Number of histogram bins. Defaults to 256.

    Returns:
        float: Entropy in nats, or nan if there are no voxels to measure.
    """
    volume = np.asarray(volume)
    values = volume[np.asarray(mask) > 0] if mask is not None else volume.ravel()
    if values.size == 0:
        return float("nan")

    counts, _ = np.histogram(values, bins=bins)
    counts = counts[counts > 0]
    if counts.size == 0:
        return float("nan")

    # stats.entropy normalises the counts into a probability distribution.
    return float(stats.entropy(counts))
