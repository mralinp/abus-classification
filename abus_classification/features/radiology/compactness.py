import numpy as np


def compactness(perimeter, area):
    return (perimeter ** 2) / (4 * np.pi * area)

def calculate_perimeter(lesion_boundary):
    # Calculate the perimeter of the lesion boundary
    return len(lesion_boundary)

def calculate_area(lesion_pixels):
    # Calculate the area of the lesion (number of pixels in the lesion)
    return len(lesion_pixels)
