import numpy as np
import torch
from abus_classification import transformers

def test_transpose():
    
    x = np.zeros((16,32,64), dtype=np.uint8)
    
    transformer = transformers.TransposeTransformer((2,1,0))
    
    res = transformer(x)
    
    assert res.shape[0] == 64
    assert res.shape[1] == 32
    assert res.shape[2] == 16
    
    
def test_resize():
    
    data = np.zeros((256,256,256), dtype=np.uint8)
    transform = transformers.Resize3DTransformer(target_size=(128,128,512))
    data = transform(data)
    x, y, z = data.shape
    
    assert x == 128
    assert y == 128
    assert z == 512
    
    
def test_slice():
    
    data = np.zeros((256,256,256), dtype=np.uint8)
    transformer = transformers.SliceTransformer(10, anchor='middle')
    data = transformer(data)
    _,_,depth = data.shape
    
    assert depth == 10
    
def test_to_tensor():
    
    data = np.zeros((256,256,256), dtype=np.uint8)
    transformer = transformers.TensorTransformer()
    data = transformer(data)
    
    assert isinstance(data, torch.Tensor)