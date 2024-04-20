import numpy as np
from abus_classification import utils


def test_signature():
    
    x = np.array([[0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,1,0,0,0,0,1,0,0],
                  [0,0,1,0,0,0,0,1,0,0],
                  [0,0,1,0,0,0,0,1,0,0],
                  [0,0,1,0,0,0,0,1,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0]], dtype=np.uint8) * 255
    
    sig = utils.features.signature(x, res=1)
    
    print(sig)
    