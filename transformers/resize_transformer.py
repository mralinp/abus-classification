import cv2
import numpy as np
from .transformer import Transformer


class Resize3DTransformer(Transformer):
    
    def __init__(self, target_size: tuple):
        self.size = target_size
        
    def transform(self, data, mask):
        n_slices = data.shape[2]
        resized_data = np.zeros(self.size, dtype=np.float32)
        resized_mask = np.zeros(self.size, dtype=np.float32)
        img_size = (self.size[0], self.size[1])
        
        for i in range(n_slices):
            data_sli = data[:,:,i]
            data_sli = cv2.resize(data_sli, img_size)
            mask_sli = mask[:,:,i]
            mask_sli = cv2.resize(mask_sli, img_size)
            resized_data[:,:,i] = data_sli
            resized_mask[:,:,i] = mask_sli
        
        return resized_data, resized_mask
    
        
        
if __name__ == "__main__":
    
    data = np.zeros((256,256,256), dtype=np.uint8)
    mask = np.zeros((256,256,256), dtype=np.uint8)
    
    transform = Resize3DTransformer(target_size=(128,128,512))
    
    data, mask = transform(data, mask)
    
    print(data.shape, mask.shape)