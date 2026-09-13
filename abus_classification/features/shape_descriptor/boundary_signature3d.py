import numpy as np
from abus_classification.utils import image


def boundary_signature3d(binary_volume: np.ndarray, resolution: tuple[float, float] = (1., 1.), dif_min: int = 5) -> np.ndarray:
    '''
    Computes the radial boundary signature of a 3D shape.

    Every surface voxel is placed in an (alpha, beta) bin by two angles
    measured from the shape's centre: alpha in the (axis 0, axis 1) plane and
    beta in the (axis 0, axis 2) plane, each over the full 0-360 degree range.
    Each bin holds the mean centre-to-surface distance of its voxels; empty
    bins are 0.

    Args:
        binary_volume (ndarray): 3D binary array of the filled shape.
        resolution (tuple[float, float]): Angular bin size in degrees for
            alpha and beta respectively.
        dif_min (int): Kept for API compatibility; not used in 3D.

    Returns:
        ndarray: float32 array of shape (ceil(360 / resolution[0]),
        ceil(360 / resolution[1])).
    '''
    assert dif_min > 0
    assert resolution[0] > 0
    assert resolution[1] > 0

    n_alpha_bins = int(np.ceil(360 / resolution[0]))
    n_beta_bins = int(np.ceil(360 / resolution[1]))
    signature = np.zeros((n_alpha_bins, n_beta_bins), dtype=np.float32)

    binary = np.asarray(binary_volume) > 0
    if not binary.any():
        return signature

    center = np.argwhere(binary).mean(axis=0)
    points = image.get_surface_points(binary.astype(np.uint8))
    offsets = points - center

    # arctan2 keeps the quadrant and cannot divide by zero; the previous
    # arctan(y / x) folded everything into 0-180, leaving half the bins empty.
    alpha = np.degrees(np.arctan2(offsets[:, 1], offsets[:, 0])) % 360
    beta = np.degrees(np.arctan2(offsets[:, 2], offsets[:, 0])) % 360

    alpha_bin = np.round(alpha / resolution[0]).astype(int) % n_alpha_bins
    beta_bin = np.round(beta / resolution[1]).astype(int) % n_beta_bins

    # Distance from the centre (previously from the array origin).
    distance = np.linalg.norm(offsets, axis=1)

    sums = np.zeros_like(signature, dtype=np.float64)
    counts = np.zeros_like(signature, dtype=np.float64)
    np.add.at(sums, (alpha_bin, beta_bin), distance)
    np.add.at(counts, (alpha_bin, beta_bin), 1)

    np.divide(sums, counts, out=sums, where=counts > 0)
    return sums.astype(np.float32)
