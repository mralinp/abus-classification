import numpy as np


def posterior_acoustic_behavior(volume: np.ndarray,
                                mask: np.ndarray,
                                axis: int = 1,
                                margin: int = 3,
                                depth: int = 20) -> float:
    """
    Measure the posterior acoustic behavior behind a lesion.

    Sound passing through a lesion either loses energy, leaving the tissue
    behind it darker than its surroundings (shadowing, associated with
    malignancy), or passes more freely, leaving it brighter (enhancement,
    more often benign).

    The measure compares a slab of tissue just beyond the far edge of the
    lesion against tissue at the same depth to either side of it, and returns
    the contrast between them:

        (posterior mean - reference mean) / reference mean

    Negative values indicate shadowing, positive values enhancement, and
    values near zero no significant change. Reporting the raw contrast rather
    than a label keeps the feature usable by a classifier; see
    :func:`classify_posterior_acoustic` for the categorical reading.

    Args:
        volume (np.ndarray): The intensity volume.
        mask (np.ndarray): Binary lesion mask, same shape as `volume`.
        axis (int): The beam/depth axis along which sound travels. Defaults
            to 1, the y axis of this project's (z, y, x) ordering.
        margin (int): Voxels skipped just past the lesion, to avoid its own
            blurred edge.
        depth (int): Thickness of the slab sampled behind the lesion.

    Returns:
        float: The posterior-to-reference contrast, or nan if either region
        is empty (for example when the lesion sits at the far edge of the
        volume, leaving nothing behind it to measure).
    """
    volume = np.asarray(volume)
    mask = np.asarray(mask) > 0
    if volume.shape != mask.shape:
        raise ValueError("volume and mask must have the same shape")
    if not mask.any():
        return float("nan")

    # Footprint: where the lesion projects onto the plane orthogonal to `axis`.
    footprint = mask.any(axis=axis)
    far_edge = int(np.argwhere(mask).max(axis=0)[axis])

    start = far_edge + 1 + margin
    stop = min(start + depth, volume.shape[axis])
    if start >= stop:
        return float("nan")

    slab = np.moveaxis(volume, axis, 0)[start:stop]
    posterior = slab[:, footprint]
    reference = slab[:, ~footprint]
    if posterior.size == 0 or reference.size == 0:
        return float("nan")

    reference_mean = float(reference.mean())
    if reference_mean == 0:
        return float("nan")

    return float((posterior.mean() - reference_mean) / reference_mean)


def classify_posterior_acoustic(contrast: float, threshold: float = 0.1) -> str:
    """
    Turn the contrast from :func:`posterior_acoustic_behavior` into a label.

    Args:
        contrast (float): The value returned by `posterior_acoustic_behavior`.
        threshold (float): Magnitude below which the change is called
            insignificant.

    Returns:
        str: "shadowing", "enhancement", "no significant change", or "unknown"
        when the contrast could not be measured.
    """
    if contrast is None or np.isnan(contrast):
        return "unknown"
    if contrast <= -threshold:
        return "shadowing"
    if contrast >= threshold:
        return "enhancement"
    return "no significant change"
