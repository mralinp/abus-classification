import numpy as np
from .transformer import Transformer


class SliceTransformer(Transformer):
    
    def __init__(self, offset: int, anchor='middle'):
        self.offset = offset
        self.anchor = anchor
        
        
    def transform(self, data: np, mask: np):
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
        new_mask = np.array(mask[:,:,start:end])
        
        return new_data, new_mask
    
    
if __name__ == "__main__":
    
    data = np.zeros((256,256,256), dtype=np.uint8)
    mask = np.zeros((256,256,256), dtype=np.uint8)
    
    transform = SliceTransformer(offset=20)
    
    data, mask = transform(data, mask)
    
    print(data.shape, mask.shape)