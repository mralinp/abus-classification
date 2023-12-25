import cv2
import numpy as np


def get_boundary(binary_img):
    uint_img = binary_img.astype(np.uint8)
    return cv2.Canny(uint_img, 0, 1)


def find_shape_center(binary_img):
    nonzero_cords = np.nonzero(binary_img)
    x_cords = nonzero_cords[0]
    y_cords = nonzero_cords[1]
    return x_cords.sum()//len(x_cords), y_cords.sum()//len(y_cords)


def signature(binary_image, res=2):
    x0, y0 = find_shape_center(binary_image)
    c = x0 - y0
    boundary = get_boundary(binary_image)
    cords_x, cords_y = np.nonzero(boundary)
    boundary_cords = list(zip(cords_x, cords_y))
    print(boundary_cords)
    signature = np.zeros([360//res], dtype=np.float32)
    for theta in range(0, 360, res):
        slop = np.tan(theta)
        # m.x + c = y => c = tan(theta).x0-y0
        c = slop*x0-y0
        for x,y in boundary_cords:
            y_pred = slop*x + c
            if y_pred - y < 0.01:
                signature[theta//res] = np.sqrt((x0-x)**2 + (y0-y)**2)
    return signature