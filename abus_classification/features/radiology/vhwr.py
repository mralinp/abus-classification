import numpy as np


def height_to_width_ratio(x: np):
    '''
    Calculates the ratio of the tumor width to its height
    
    param x: The 3D tumor
    returns float
    '''
    w,h,d = x.shape
    return w / h
