import numpy as np
from skimage import morphology


def margin_contrast(volume: np.ndarray, mask: np.ndarray) -> float:
    '''
    Calculates the Margin Intensity feature according to [Tao et al. 2013]

    The mean intensity of a one-voxel shell just inside the lesion boundary
    minus that of a one-voxel shell just outside it. A sharp, well-defined
    margin gives a large magnitude; an indistinct one gives a value near 0.

    Args:
        volume (ndarray):   Original 2D/3D lesion image
        mask (ndarray):     lesion segmentation (mask)

    Returns:
        float: The difference of inner and outer margin intensity, or nan if
        either shell is empty (for example when the lesion fills the array,
        leaving no voxels outside it).
    '''
    volume = np.asarray(volume)
    mask = np.asarray(mask) > 0
    if volume.shape != mask.shape:
        raise ValueError("volume and mask must have the same shape")

    structuring_element = morphology.disk(1) if volume.ndim == 2 else morphology.ball(1)

    inner_margin = mask & ~morphology.erosion(mask, structuring_element)
    outer_margin = morphology.dilation(mask, structuring_element) & ~mask

    if not inner_margin.any() or not outer_margin.any():
        return float("nan")

    return float(volume[inner_margin].mean() - volume[outer_margin].mean())
