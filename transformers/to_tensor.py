
import numpy as np
import torch


class ToTensorTransformer:
    def __call__(self, sample):
        x, m = sample
        x = torch.from_numpy(x)
        y = torch.from_numpy(m)
        
        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        y = y.unsqueeze(0).permute(0, 3, 1, 2)
        
        return x, y