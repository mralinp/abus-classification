import torch
import pandas as pd
import scipy.io

import os


class ABUS(torch.utils.data.Dataset):
    
    def __init__(self, path='./data/abus', transforms=None):
        
        self.root_path = path
        self.transforms = transforms
        self.list_data = os.listdir(f"{self.root_path}/volumes")
        self.labels = pd.read_excel(f"{self.root_path}/labels.xlsx")
    
    def __len__(self):
        
        return len(self.list_data)
    
    def __getitem__(self, index):
        path = self.list_data[index]
        volume = scipy.io.loadmat(f"{self.root_path}/volumes/{path}")['volume']
        mask = scipy.io.loadmat(f"{self.root_path}/masks/{path}")['bw']
        name = path.replace('.mat', '')

        desired_row = self.labels[self.labels['ImageID'] == name]

        if desired_row.empty:
            desired_row = self.labels[self.labels['ImageID'] == f"{name}_1"]

        label = desired_row["Class"].values[0]
        
        if self.transforms is not None:
            pass
        
        return volume, mask, label

if __name__ == '__main__':
    dataset = ABUS(path="./data/abus")
    print(len(dataset))
    x,y,l = dataset[0]
    print(x.shape, y.shape, l)