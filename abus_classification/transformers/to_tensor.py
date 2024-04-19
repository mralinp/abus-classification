
import numpy as np
import torch


class ToTensorTransformer:
    def __call__(self, sample):
        x, y = sample
        x = x.transpose((2,0,1))
        y = y.transpose((2,0,1))
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        return x, y