import torch
import numpy as np
import os


class Tumors(torch.utils.data.Dataset):
    def __init__(self, path="./data/tdsc/tumors_3d/z", transforms=None):
        
        self.root_path = path
        self.transforms = transforms
        self.data_list = [(data_path, 0) for data_path in os.listdir(f"{path}/data/0")] + [(data_path, 1) for data_path in os.listdir(f"{path}/data/1")]

    def __getitem__(self, index):
        
        path, y = self.data_list[index]
        v = np.load(f"{self.root_path}/data/{y}/{path}")
        m = np.load(f"{self.root_path}/mask/{y}/{path}")
        
        if self.transforms:
            for transform in self.transforms:
                m, v = transform(data=v,mask=m)
                
        return v,m,y

    
    def __len__(self):
        return len(self.data_list)
    
if __name__ == '__main__':
    data = Tumors()
    v,m,y = data[0]
    
    print(v.shape, m.shape, y)