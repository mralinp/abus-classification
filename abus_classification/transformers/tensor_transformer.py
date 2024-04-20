import torch
import numpy as np
from .transformer import Transformer


class TensorTransformer(Transformer):
    
    def turn_to_tensor(self, data):
        return torch.from_numpy(data)
    
    def transform(self, *inputs):
        
        assert len(inputs) > 0
        res = []
        for data in inputs:
            res.append(self.turn_to_tensor(data))
            
        res = tuple(res) if len(res) > 1 else res.pop()
        
        return res