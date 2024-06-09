import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy import ndimage as ndi


def fractal_dimension(skeleton):
    # Compute fractal dimension using box-counting method
    L = skeleton.shape[0]
    num_boxes = np.floor(np.log(L) / np.log(2))
    counts = []

    for i in range(int(num_boxes)):
        s = int(2 ** i)
        boxes = ndi.labeled_comprehension(skeleton, np.zeros(skeleton.shape), np.arange(1, skeleton.max() + 1), np.sum, int, None, (s, s))
        counts.append(np.count_nonzero(boxes))

    counts = np.array(counts)
    log_counts = np.log(counts)
    log_scales = np.log(2 ** np.arange(int(num_boxes)))
    slope, _, _, _, _ = np.linalg.lstsq(log_scales.reshape(-1, 1), log_counts, rcond=None)

    return -slope[0]


def spiculation(x, m):
    # Convert segmented mask to binary image
    m_binary = np.uint8(m > 0)

    # Skeletonize the binary mask
    skeleton = skeletonize(m_binary)

    # Detect edges of the segmented tumor
    edges = cv2.Canny(np.uint8(x), 30, 70)

    # Calculate spiculation length
    spiculation_length = np.sum(skeleton)

    # Calculate spiculation angle
    dx, dy = np.gradient(skeleton)
    angles = np.arctan2(dy, dx) * (180 / np.pi)
    spiculation_angles = np.unique(angles)

    # Calculate spiculation density
    spiculation_density = np.sum(skeleton) / np.sum(x)

    # Calculate spiculation complexity (e.g., using fractal dimension)
    fractal_dimension_feature = fractal_dimension(skeleton)

    return {
        'spiculation_length': spiculation_length,
        'spiculation_angles': spiculation_angles,
        'spiculation_density': spiculation_density,
        'fractal_dimension': fractal_dimension_feature
    }
