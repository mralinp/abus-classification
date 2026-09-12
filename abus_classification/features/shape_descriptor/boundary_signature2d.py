import numpy as np


def boundary_signature2d(boundary_image: np.ndarray, resolution: float = 1., dif_min: int = 5) -> np.ndarray:
    '''
    Computes the radial boundary signature of a 2D shape.

    For each angle theta = 0, resolution, 2*resolution, ... a ray is cast from
    the centre of the boundary points, and the signature records the distance
    to the boundary point lying closest to that ray, on the side it points
    towards. Points further than `dif_min` pixels from the ray are ignored;
    angles with no such point get 0.

    Angles are measured in (row, column) coordinates, so theta = 0 points down
    the rows and theta = 90 along the columns.

    Args:
        boundary_image (ndarray): 2D array whose non-zero pixels are the boundary.
        resolution (float): Angular step in degrees.
        dif_min (int): Maximum perpendicular distance, in pixels, between a
            boundary point and the ray for the point to count.

    Returns:
        ndarray: float32 array of length 360 / resolution.
    '''
    assert dif_min >= 0
    assert resolution > 0

    n_bins = int(round(360 / resolution))
    signature = np.zeros(n_bins, dtype=np.float32)

    points = np.argwhere(np.asarray(boundary_image) > 0).astype(np.float64)
    if points.size == 0:
        return signature

    offsets = points - points.mean(axis=0)
    distances = np.hypot(offsets[:, 0], offsets[:, 1])

    thetas = np.radians(np.arange(n_bins) * resolution)
    directions = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)          # (bins, 2)

    along = directions @ offsets.T                                            # (bins, points)
    across = np.abs(directions[:, [0]] * offsets[:, 1] - directions[:, [1]] * offsets[:, 0])

    # Only points in front of the centre and within tolerance of the ray count.
    # The previous line-based version matched points on both sides of the
    # centre and used a horizontal line for 90 and 270 degrees.
    candidate = (along >= 0) & (across < dif_min)
    penalty = np.where(candidate, across, np.inf)

    best = penalty.argmin(axis=1)
    found = np.isfinite(penalty[np.arange(n_bins), best])
    signature[found] = distances[best[found]]
    return signature
