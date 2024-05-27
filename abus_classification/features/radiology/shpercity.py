import numpy as np


def sphericity(volume, surface_area):
    return (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area

def calculate_volume(lesion_voxels):
    return len(lesion_voxels)

def calculate_surface_area(lesion_voxels):
    # A placeholder function. You would need to calculate the actual surface area.
    # This could be done using a mesh reconstruction method, for example.
    return len(lesion_voxels) ** (2/3)
