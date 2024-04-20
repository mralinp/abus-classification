import numpy as np
from .transformer import Transformer


class TransposeTransformer(Transformer):
    
    def __init__(self, shape: tuple):
        self.shape = shape
    
    def transform(self, *args):
        assert len(args) > 0
        res = []
        for arg in args:
           res.append(np.transpose(arg, self.shape)) 
        
        res = tuple(res) if len(res) > 1 else res.pop()
        
        return res