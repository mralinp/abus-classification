import numpy as np
import cv2
import matplotlib.pyplot as plt
from . import math

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


def find_bbx(slice_img):
    '''
    Takes binary image containing a tumor mask inside and returns the bounding box containing that tumor.
    
    Parameters:    
    - slice_image: binary image containing tumor mask inside
    
    Returns:
    - tuple(x, y, width, height)
    '''    
    non_zeros = cv2.findNonZero(slice_img)
    return cv2.boundingRect(non_zeros)



def rotate_image(x, degree):
    height, width = x.shape
    rotation_matrix = cv2.getRotationMatrix2D(center=(width // 2, height // 2), angle=degree, scale=1)
    return cv2.warpAffine(src=x, M=rotation_matrix, dsize=(width, height))


def get_boundary(binary_img):
    uint_img = binary_img.astype(np.uint8)
    return cv2.Canny(uint_img, 0, 1)


def find_shape_center(binary_img):
    nonzero_cords = np.nonzero(binary_img)
    x_cords = nonzero_cords[0]
    y_cords = nonzero_cords[1]
    return x_cords.sum()//len(x_cords), y_cords.sum()//len(y_cords)


def rotation_invariant(x):
    w, h = x.shape
    w = max(w,h)
    x = zero_pad_resize(x, size=(w,w))
    boundary = get_boundary(x)
    cords_x, cords_y = np.nonzero(boundary)
    boundary_points = list(zip(cords_x, cords_y))
    slop = math.calculate_slope(math.find_farthest(boundary_points))
    res = rotate_image(x, np.degrees(np.arctan(slop)))
    h, _ = x.shape
    a = res[:h//2,:].sum()
    b = res[h//2:,:].sum()
    
    if a > b:
        res = rotate_image(res, 180)
    
    return res

def crop(img, center, size=(224,224)):
    cx,cy = center
    w,h = size
    return img[cx-w//2:cx+w//2,cy-h//2:cy+h//2]


def find_tumors(mask: np) -> np:
    '''
    Finds all bounding boxes in volume containing tumors
    
    parameters:
    - mask: numpy array with shape (x,y,d)
    
    returns:
    - numpy array with (None, [x,y,w,h,d])
    '''
    
    _, _, depth = mask.shape
    bbxs = []
    for d in range(depth):
        sli = mask[:,:,d]
        if sli.max() != 0:
            *bbx, = find_bbx(sli)
            bbx.append(d)
            bbxs += [bbx]
    return np.array(bbxs)


def find_largest_bounding_box(bbxs: list):
    
    xl,yl = 1000, 1000
    xh,yh = 0, 0
    
    for bbx in bbxs:
        
        x,y,w,h,_ = bbx
        xl = min(xl, x)
        yl = min(yl, y)
        xh = max(xh, x+w)
        yh = max(yh, y+h)
        
    return (xl,yl), (xh,yh)
