import os

import torch
import numpy as np


class TDSCTexture(torch.utils.data.Dataset):
    
    def __init__(self, path="./data/tdsc/texture"):
        self.path = path
        self.data_list = [(p, 0) for p in os.listdir(f"{path}/malignant")] + [(p, 1) for p in os.listdir(f"{path}/benign")]
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        path, label = self.data_list[index]
        directory = 'malignant' if label == 0 else 'benign'
        data = np.load(f"{self.path}/{directory}/{path}")
        return data, label
        

if __name__ == "__main__":
    data = TDSCTexture(path="./data/tdsc/texture")
    d, l = data[0]
    print(d.shape, l)