import numpy as np
from abus_classification import utils


def test_boundary_signature_2d():
    
    x = np.array([[1,1,1,1,1],
                  [1,0,0,0,1],
                  [1,0,0,0,1],
                  [1,0,0,0,1],
                  [1,1,1,1,1]], dtype=np.uint8) * 255
    y = np.array([[2.,2.828427,2.,2.828427,2.,2.828427,2.,2.828427]], dtype=np.float32)
    
    sig = utils.features.boundary_signature_2d(x, resolution=45)
    
    print(sig)
    
    assert np.equal(sig,y).all()
    
def test_boundary_signature_3d():
    x = np.zeros((5,5,5), dtype=np.uint8)
    x[1:4,1:4,1:4] = np.ones((1,1,1), dtype=np.uint8)
    
    res = utils.features.boundary_signature_3d(x, resolution=(15,15))
    
    print(res)