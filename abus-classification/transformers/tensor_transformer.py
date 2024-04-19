import torch
import numpy as np
from .transformer import Transformer


class TensorTransformer(Transformer):
    
    def transform(self, data, mask):
        data_tensor = torch.from_numpy(data)
        mask_tensor = torch.from_numpy(mask)
        return data_tensor, mask_tensor
    
    
if __name__ == "__main__":
    
    data = np.zeros((256,256,3), dtype=np.uint8)
    mask = np.zeros((256,256,3), dtype=np.uint8)
    
    transform = TensorTransformer()
    
    data, mask = transform(data, mask)
    
    print(data.shape, mask.shape)