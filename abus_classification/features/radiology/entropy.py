from skimage.measure import shannon_entropy


def entropy(lesion_pixels):
    return shannon_entropy(lesion_pixels)
