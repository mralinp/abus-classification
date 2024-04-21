import numpy as np
from .transformer import Transformer


class SliceTransformer(Transformer):
    
    def __init__(self, start:int, end:int):
        self.start = start
        self.end = end
        
    def make_slice(self, data):
        return np.array(data[:,:,self.start:self.end])
        
    def transform(self, *inputs):
        
        assert len(inputs) > 0
        res = []
        
        for data in inputs:
            res.append(self.make_slice(data))
        
        res = tuple(res) if len(res) > 1 else res.pop()
        
        return res