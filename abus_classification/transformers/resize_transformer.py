import cv2
import numpy as np
from .transformer import Transformer


class Resize3DTransformer(Transformer):
    
    def __init__(self, target_size: tuple):
        self.size = target_size
        
    def resize(self, data):
        n_slices = data.shape[2]
        resized_data = np.zeros(self.size, dtype=np.float32)
        img_size = (self.size[0], self.size[1])
        
        for i in range(n_slices):
            data_sli = data[:,:,i]
            data_sli = cv2.resize(data_sli, img_size)
            resized_data[:,:,i] = data_sli
        
        return resized_data
    
    def transform(self, *inputs):
        
        assert len(inputs) > 0
        
        res = []
        
        for data in inputs:
            res.append(self.resize(data))
        
        res = tuple(res) if len(res) > 1 else res.pop()
        
        return res