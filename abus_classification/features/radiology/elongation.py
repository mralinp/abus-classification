import numpy as np


def _principal_variances(mask: np.ndarray) -> np.ndarray:
    """Variances of the lesion voxel coordinates along its principal axes, largest first."""
    points = np.argwhere(np.asarray(mask) > 0).astype(np.float64)
    if points.shape[0] < 2:
        return np.full(np.asarray(mask).ndim, np.nan)
    return np.sort(np.linalg.eigvalsh(np.cov(points, rowvar=False)))[::-1]


def elongation(mask: np.ndarray) -> float:
    """
    Calculate the elongation of a lesion from its principal axes.

    Follows the PyRadiomics definition, sqrt(lambda_minor / lambda_major),
    where the lambdas are the two largest variances of the voxel coordinates
    along the lesion's principal axes. The result lies in (0, 1]: 1 for a
    lesion as wide as it is long, towards 0 for a needle-like one. Because
    the axes come from the lesion itself, the value does not depend on how
    the lesion happens to be oriented in the volume.

    Args:
        mask (np.ndarray): A 2D or 3D binary lesion mask.

    Returns:
        float: The elongation, or nan for masks with fewer than two voxels.
    """
    variances = _principal_variances(mask)
    if np.isnan(variances[0]) or variances[0] <= 0:
        return float("nan")
    return float(np.sqrt(max(variances[1], 0.0) / variances[0]))


def flatness(mask: np.ndarray) -> float:
    """
    Calculate the flatness of a 3D lesion from its principal axes.

    Follows the PyRadiomics definition, sqrt(lambda_least / lambda_major).
    The result lies in (0, 1]: 1 for a lesion with no thin direction, towards
    0 for a flat, plate-like one.

    Args:
        mask (np.ndarray): A 3D binary lesion mask.

    Returns:
        float: The flatness, or nan for masks with fewer than two voxels.
    """
    assert np.asarray(mask).ndim == 3, "The mask should be a 3D ndarray."
    variances = _principal_variances(mask)
    if np.isnan(variances[0]) or variances[0] <= 0:
        return float("nan")
    return float(np.sqrt(max(variances[2], 0.0) / variances[0]))


def bounding_box_fill(mask: np.ndarray) -> float:
    """
    Calculate the fraction of the lesion's bounding box that the lesion fills.

    Also known as extent. A solid, box-like lesion approaches 1; an irregular
    or branching one leaves most of its box empty and scores low.

    Args:
        mask (np.ndarray): A binary lesion mask of any dimensionality.

    Returns:
        float: Lesion voxels divided by bounding-box voxels, or nan if empty.
    """
    mask = np.asarray(mask) > 0
    points = np.argwhere(mask)
    if points.size == 0:
        return float("nan")
    box = np.prod(np.ptp(points, axis=0) + 1)
    return float(mask.sum() / box)
