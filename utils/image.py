import numpy as np
import cv2
import matplotlib.pyplot as plt


def zero_pad_resize(img, size=(224,224)):
    res = np.zeros(size, dtype=np.float32)
    cx, cy = size[0]//2, size[1]//2
    w,h = img.shape
    if w > size[0] or h > size[1]:
        raise Exception(f"Cant resize with zero padding from origin with shape {img.shape} to size {size}")
    res[cx-w//2:cx+w//2 + w%2, cy-h//2:cy+h//2 + h%2] = img
    return res


def find_squer_from_rect(bbx):
    x, y, w, h, d = bbx
    x, y = x+w//2, y+h//2
    w = max(w,h)
    return (x - w//2, y - w//2, w, w, d)


def show_image_mask_bbx(img, mask, bbx):
    msk = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    msk = cv2.rectangle(msk, (bbx[0], bbx[1]), (bbx[0] + bbx[2], bbx[1] + bbx[3]), (0, 255, 0), 2)
    plt.figure()
    plt.subplot(1,2,1)
    plt.title("Image")
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.title("Mask")
    plt.imshow(msk)
    plt.show()


def find_bbx(mask):    
    non_zeros = cv2.findNonZero(mask)
    return cv2.boundingRect(non_zeros)

def subsample_image(img, size=(224,224), stride=1):
    w,h = img.shape
    res = []
    for i in range(0, w-size[0], stride):
        for j in range(0, h-size[1], stride):
            res += [img[i:i + size[0], j:j+size[1]]]
    return np.array(res, dtype=np.float32)
