import numpy as np
from skimage import morphology


def fractal_dimension(binary: np.ndarray) -> float:
    """
    Estimate the Minkowski-Bouligand (box-counting) dimension of a binary set.

    The array is padded up to a power-of-two cube, then covered with boxes of
    side 1, 2, 4, ... and the occupied boxes counted at each scale. The
    dimension is the slope of log(count) against log(1 / side).

    Args:
        binary (np.ndarray): Binary array of any dimensionality.

    Returns:
        float: The box-counting dimension, or nan if the set is empty.
    """
    binary = np.asarray(binary) > 0
    if not binary.any():
        return float("nan")

    ndim = binary.ndim
    side = 2 ** int(np.ceil(np.log2(max(binary.shape))))
    padded = np.zeros((side,) * ndim, dtype=bool)
    padded[tuple(slice(0, s) for s in binary.shape)] = binary

    scales, counts = [], []
    for box in (2 ** np.arange(int(np.log2(side)))):
        box = int(box)
        blocks = padded.reshape(
            [dim for _ in range(ndim) for dim in (side // box, box)]
        )
        occupied = blocks.sum(axis=tuple(range(1, 2 * ndim, 2)))
        scales.append(box)
        counts.append(int(np.count_nonzero(occupied)))

    if len(scales) < 2:
        return float("nan")

    slope, _ = np.polyfit(np.log(1.0 / np.asarray(scales, dtype=float)), np.log(counts), 1)
    return float(slope)


def spiculation(mask: np.ndarray) -> dict:
    """
    Quantify how spiculated (spiky and irregular) a lesion margin is.

    Two complementary measures are returned:

    - `boundary_fractal_dimension` - the box-counting dimension of the lesion
      surface. A smooth surface sits near the topological dimension of the
      boundary; convoluted, spiculated margins push it higher.
    - `radial_distance_deviation` - the standard deviation of centroid-to-
      boundary distances divided by their mean. It is 0 for a perfect sphere
      or circle and grows as spicules stretch the margin unevenly.
    - `boundary_voxels` - the size of the extracted boundary, for reference.

    Args:
        mask (np.ndarray): A 2D or 3D binary array containing the lesion mask.

    Returns:
        dict: The three measures above, as floats.
    """
    mask = np.asarray(mask) > 0
    empty = {
        "boundary_fractal_dimension": float("nan"),
        "radial_distance_deviation": float("nan"),
        "boundary_voxels": 0.0,
    }
    if not mask.any():
        return empty

    boundary = mask & ~morphology.binary_erosion(mask)
    points = np.argwhere(boundary)
    if points.size == 0:
        return empty

    centroid = np.argwhere(mask).mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    mean_distance = distances.mean()

    return {
        "boundary_fractal_dimension": fractal_dimension(boundary),
        "radial_distance_deviation": (
            float(distances.std() / mean_distance) if mean_distance else float("nan")
        ),
        "boundary_voxels": float(points.shape[0]),
    }
