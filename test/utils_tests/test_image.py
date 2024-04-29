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
    
    
def test_find_center():    
    x = np.array([[0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,1,1,1,1,0,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,0,1,1,1,1,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0]], dtype=np.uint8) * 255
    
    center_point = image.find_shape_center(x)
        
    assert center_point.shape[0] == 2 
    assert center_point[1] == 4.5
    assert center_point[0] == 4.5

def test_get_boundary():
    
    x = np.array([[0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0]], dtype=np.uint8)
    
    y = np.array([[0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,1,0,0,0,0,1,0,0],
                  [0,0,1,0,0,0,0,1,0,0],
                  [0,0,1,0,0,0,0,1,0,0],
                  [0,0,1,0,0,0,0,1,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0]], dtype=np.uint8)
    
    boundary = image.get_boundary(x)
    
    print(boundary)
    
    assert np.equal(boundary, y).all()
    
    
def test_get_surface_points():
    x = np.array([[0,0,0,0,0],
                  [0,1,1,1,0],
                  [0,1,1,1,0],
                  [0,1,1,1,0],
                  [0,0,0,0,0]], dtype=np.uint8)
    
    y = np.array([[1, 1],
                  [1, 2],
                  [1, 3],
                  [2, 1],
                  [2, 3],
                  [3, 1],
                  [3, 2],
                  [3, 3]])

    points = image.get_surface_points(x)
    
    assert (points == y).all()