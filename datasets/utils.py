import cv2
import numpy as np

def zero_pad_resize(img: np, size:tuple=(512,512)) -> np:
    res = np.zeros(size, dtype=np.float32)
    cx, cy = size[0]//2, size[1]//2
    w,h = img.shape
    if w > size[0] or h > size[1]:
        raise Exception(f"Cant resize with zero padding from origin with shape {img.shape} to size {size}")
    res[cx-w//2:cx+w//2 + w%2, cy-h//2:cy+h//2 + h%2] = img
    return res