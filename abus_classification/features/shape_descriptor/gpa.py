import cv2
import numpy as np


def gps(image):
    '''
    Calculates the Global Point Signature of the given 3D-mesh point cloud of the tumor according to [Rustamov et al. 2007]
    
    Args:
        point_cloud (trimesh):  The triangular mesh of the given lesion
    
    Returns:
        list of gps descriptors for each point in given the 3D-mesh 
    '''
    
    
    # Convert the image to grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # Detect corners using the FAST corner detector 
    corners = cv2FAST(gray, None, 15, 3, 9) 
    # Compute the SIFT descriptor for each corner 
    sift = cv2.SIFT_create() 
    keypoints = [cv2.KeyPoint(x, y, 10) for x, y in corners] 
    descriptors = [sift.compute(gray, [kpt])[0] for kpt in keypoints] 
    # Compute the GPS by concatenating the descriptors 
    gps = np.concatenate(descriptors, axis=0) 
    
    return gps

# Load an image
image = cv2.imread('image.jpg')
# Compute the GPS
gps = compute_gps(image)
# Print the GPS
print(gps)