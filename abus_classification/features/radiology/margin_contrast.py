import numpy as np
from scipy import ndimage
from skimage import morphology


def margin_contrast(volume: np.ndarray, mask: np.ndarray, spacing=None, width_mm: float = 1.2) -> float:
    '''
    Calculates the margin contrast feature of Tan et al. (2012).

    The mean intensity of a shell just inside the lesion boundary minus that of
    a shell just outside it. A sharp, well-defined margin gives a large
    magnitude; an indistinct one gives a value near 0.

    Without `spacing` both shells are one voxel thick. With `spacing` they hold
    the voxels within `width_mm` of the boundary (Tan uses 1.2 mm), measured in
    physical units, so the shells are equally thick in every direction on
    anisotropic data.

    Args:
        volume (ndarray):   Original 2D/3D lesion image
        mask (ndarray):     lesion segmentation (mask)
        spacing (tuple, optional): Physical voxel size along each axis.
        width_mm (float):   Shell thickness when `spacing` is given.

    Returns:
        float: The difference of inner and outer margin intensity, or nan if
        either shell is empty (for example when the lesion fills the array,
        leaving no voxels outside it).
    '''
    volume = np.asarray(volume)
    mask = np.asarray(mask) > 0
    if volume.shape != mask.shape:
        raise ValueError("volume and mask must have the same shape")
    if not mask.any():
        return float("nan")

    if spacing is None:
        structuring_element = morphology.disk(1) if volume.ndim == 2 else morphology.ball(1)
        inner_margin = mask & ~morphology.erosion(mask, structuring_element)
        outer_margin = morphology.dilation(mask, structuring_element) & ~mask
    else:
        sampling = tuple(float(s) for s in spacing)
        inner_margin = mask & (ndimage.distance_transform_edt(mask, sampling=sampling) <= width_mm)
        outer_margin = (~mask & (ndimage.distance_transform_edt(~mask, sampling=sampling) <= width_mm)
                        if not mask.all() else np.zeros_like(mask))

    if not inner_margin.any() or not outer_margin.any():
        return float("nan")

    return float(volume[inner_margin].mean() - volume[outer_margin].mean())
