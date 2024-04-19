import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


# Function to calculate GLCM features for a given image
def glcm(img, distances:list=[1], angles:list=[0]):
    glcm = graycomatrix(img, distances, angles, symmetric=True, normed=True)
    
    # Calculate GLCM properties
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    
    # Return the calculated features
    return {
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation,
    }
