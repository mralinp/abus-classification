import os
import random
import torch
import numpy as np
from PIL import Image


class TDSCForDetection(torch.utils.data.Dataset):
    
    def __init__(self, path="./data/tdsc/detection", train=True, transforms=None):
        self.path = path
        self.transform = transforms
        self.data_list = [(data_path, 0) for data_path in os.listdir(f"{path}/0")] + [(data_path,1) for data_path in os.listdir(f"{path}/1")]
        # After building the list, it should be shuffled
        random.shuffle(self.data_list)
    
    def __getitem__(self, index):
        data_path, label = self.data_list[index]
        x = np.array(Image.open(f"{self.path}/{label}/{data_path}").convert("L"))
        if self.transform is not None:
            x = self.transform(image=x)["image"]
        return x, label
    
    def __len__(self):
        return len(self.data_list)
    