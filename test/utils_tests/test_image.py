import numpy as np
from abus_classification.utils import image


def test_zero_pad_resize():
    x = np.ones([2,2], dtype=np.uint8)
    y = np.array([[0,0,0,0],
                  [0,1,1,0],
                  [0,1,1,0],
                  [0,0,0,0]], dtype=np.uint8)
    res = image.zero_pad_resize(x, size=(4,4))
    assert np.equal(res,y).all()
    
    
def test_rotate_image():
    x = np.array([[0,0,1],
                  [0,1,0], 
                  [1,0,0]], dtype=np.uint8)
    
    y = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,1]], dtype=np.uint8)

    res = image.rotate_image(x, 90)
    assert np.equal(res, y).all()
    
    
def test_find_bbx():
    x = np.array([[0,0,0,0,0,0,0,0,0,0], 
                  [0,0,0,0,0,0,0,0,0,0], 
                  [0,0,0,0,0,0,0,0,0,0], 
                  [0,0,1,1,1,1,1,1,0,0], 
                  [0,0,1,1,1,1,1,1,0,0], 
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0]], dtype=np.uint8)
    
    a,b,w,h = image.find_bbx(x)
    
    assert a == 2 and b == 3 and w == 6 and h == 4
    
    
