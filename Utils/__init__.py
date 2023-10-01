import cv2
import numpy as np


def dilation(src: np, size: tuple) -> np:
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41), (20, 20))
    return cv2.dilate(src, element)