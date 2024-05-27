import numpy as np
from skimage import morphology


def margin_contrast(volume:np, mask:np) -> float:
    '''
    Calculates the Margin Intensity feature according to the [Tao et .al 2013]
    
    Args:
        volume (ndarray):   Original 2D/3D lesion image
        mask (ndarray):     lesion segmentation (mask) 
    
    Returns:
        The difference of inner and outer margin intensity
    '''
    
    # Ensure the mask is binary
    mask = mask > 0
    
    # Determine if the data is 2D or 3D
    is_2d = volume.ndim == 2

    # Choose the appropriate structuring element
    if is_2d:
        structuring_element = morphology.disk(1)
    else:
        structuring_element = morphology.ball(1)

    # Find the inner margin (erosion)
    inner_margin = mask & ~morphology.erosion(mask, structuring_element)

    # Find the outer margin (dilation)
    outer_margin = morphology.dilation(mask, structuring_element) & ~mask

    # Get the intensity values at the inner and outer margins
    inner_margin_intensity_values = volume[inner_margin]
    outer_margin_intensity_values = volume[outer_margin]

    # Calculate the average intensity for each margin
    mean_inner_margin_intensity = np.mean(inner_margin_intensity_values)
    mean_outer_margin_intensity = np.mean(outer_margin_intensity_values)

    # Calculate the difference in mean intensity
    margin_intensity_difference = mean_inner_margin_intensity - mean_outer_margin_intensity

    return margin_intensity_difference