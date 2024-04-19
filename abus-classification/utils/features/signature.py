import cv2
import numpy as np
from .. import image

def signature(binary_image, res=1):
    binary_image = image.rotation_invariant(binary_image)
    x0, y0 = image.find_shape_center(binary_image)
    c = x0 - y0
    boundary = image.get_boundary(binary_image)
    cords_x, cords_y = np.nonzero(boundary)
    boundary_cords = list(zip(cords_x, cords_y))
    signature = np.zeros([360//res], dtype=np.float32)
    for theta in range(0, 360, res):
        slop = np.tan(np.radians(theta))
        # m.x + c = y => c = tan(theta).x0-y0
        c = -1*slop*x0+y0
        dif_min = 10000
        for x, y in boundary_cords:
            y_pred = slop*x + c
            dif = np.abs(y_pred - y)
            if(dif < dif_min):
                dif_min = dif
                signature[theta//res] = np.sqrt((x0-x)**2 + (y0-y)**2)
    return signature