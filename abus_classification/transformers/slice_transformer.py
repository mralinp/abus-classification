import numpy as np
from .transformer import Transformer


class SliceTransformer(Transformer):
    
    def __init__(self, offset: int, anchor='middle'):
        self.offset = offset
        self.anchor = anchor
        
    def make_slice(self, data):
        start, end = 0, 0
        if self.anchor == 'middle':
            mid = data.shape[2]//2
            offset_half = self.offset//2
            start = mid-offset_half
            end = mid+offset_half
        elif self.anchor == 'start':
            start = 0
            end = self.offset
        elif self.anchor == end:
            start = data.shape[2]-self.offset
            end = data.shape[2]
            
        new_data = np.array(data[:,:,start:end])
        return new_data
        
    def transform(self, *inputs):
        
        assert len(inputs) > 0
        res = []
        
        for data in inputs:
            res.append(self.make_slice(data))
        
        res = tuple(res) if len(res) > 1 else res.pop()
        
        return res