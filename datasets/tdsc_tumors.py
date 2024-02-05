import torch
import cv2
import numpy as np

import os
import random
from PIL import Image

from . import utils


class TDSCTumors(torch.utils.data.Dataset):
    
    def __init__(self, path: str = "./data/tdsc/tumors", transforms: list = None):
        self.path = path
        self.transforms = transforms
        self.data_list = [(data_path, 0) for data_path in os.listdir(f"{path}/data/0")] + [(data_path, 1) for data_path in os.listdir(f"{path}/data/1")]

    
    def __getitem__(self, index):
        path, y = self.data_list[index]
        slices_list = os.listdir(f"{self.path}/data/{y}/{path}")
        slices_list.sort()
        imgs = []
        msks = []
        for slice_name in slices_list:
            path_to_img = f"{self.path}/data/{y}/{path}/{slice_name}"
            path_to_mask = f"{self.path}/mask/{y}/{path}/{slice_name}"
            imgs += [np.array(Image.open(path_to_img).convert("L"))]
            msks += [np.array(Image.open(path_to_mask).convert("L"))]
        
        for i in range(len(imgs)):
            imgs[i] = utils.zero_pad_resize(imgs[i], (512, 512))
            msks[i] = utils.zero_pad_resize(msks[i], (512,512))
            
        return np.array(imgs, dtype=np.float32)/255, np.array(msks, dtype=np.float32)/255, y
    
    def __len__(self):
        return len(self.data_list)
    
    def shuffle(self):
        random.shuffle(self.data_list)
    
    
if  __name__ == "__main__":
    data = TDSCTumors(path="../data/tdsc/tumors")
    print(len(data))
    x,m,y = data[0]
    print(x.shape, m.shape, y)
    print(x.max(), m.max())