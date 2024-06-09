import numpy as np


def variance_of_intensities(lesion_pixels):
    '''
    Calculates the variance of tumor intensities, 
    
    param lesion_pixels
    returns float 
    '''
    return np.var(lesion_pixels)
