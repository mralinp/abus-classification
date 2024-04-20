import cv2
import numpy as np
from .. import image

def boundary_signature_2d(boundary_image: np, resolution:float=1., dif_min:int=5) -> np:
    
    assert dif_min >= 0
    assert resolution >= 0
    
    x0, y0 = image.find_shape_center(boundary_image)
    cords_x, cords_y = np.nonzero(boundary_image)
    boundary_cords = list(zip(cords_x, cords_y))
    signature = np.zeros([int(360/resolution)], dtype=np.float32)
    len_signature = signature.shape[0]

    for idx in range(len_signature):
        # m*x + c = y => c = -tan(theta)*x0 + y0
        theta = idx*resolution
        slop = np.tan(np.radians(theta))
        if theta == 90 or theta == 270:
            slop = 0
        c = -1*slop*x0+y0
        best_dif = 1000
        for x, y in boundary_cords:
            y_pred = slop*x + c
            dif = np.abs(y_pred - y)
            if dif < dif_min and dif < best_dif:
                best_dif = dif
                signature[idx] = np.sqrt((x0-x)**2 + (y0-y)**2)

    return signature