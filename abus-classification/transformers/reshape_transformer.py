import numpy as np
from .transformer import Transformer


class ReshapeTransformer(Transformer):
    
    def __init__(self, shape: tuple):
        self.shape = shape
    
    def transform(self, data, mask):
        data = np.reshape(data, self.shape)
        mask = np.reshape(data, self.shape)
        
        return data, mask
    
    
if __name__ == "__main__":
    
    data = np.zeros((128,128,3), dtype=np.uint8)
    mask = np.zeros((128,128,3), dtype=np.uint8)

    transform = ReshapeTransformer(shape=(3,128,128))
    data, mask = transform(data, mask)
    
    print(data.shape)