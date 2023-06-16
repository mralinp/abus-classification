import torch
import pandas as pd
import nrrd
import os


class TDSC(torch.utils.data.Dataset):
    
    # default constructor
    def __init__(self, path_to_dataset: str = "../data/tdsc", type: str = "train") -> None:
        
        self.path_to_dataset = f"{path_to_dataset}/{type}"
        
        if not os.path.exists(path_to_dataset):
            # download dataset
            pass
        self.meta = pd.read_csv(f"{self.path_to_dataset}/labels.csv", dtype={'Case_id': int, 'Label': str, 'Data_path': str, 'Mask_path': str}).set_index('case_id')
        
    # returns item index of dataset (x, y)
    def __getitem__(self, index):
        data = self.meta.iloc[index]
        x, _ = nrrd.read(self.path_to_dataset + "/" + data.data_path.replace('\\', '/'))
        m, _ = nrrd.read(self.path_to_dataset + "/" + data.mask_path.replace('\\', '/'))
        y    = data.label   
        return x, m, y

    # returns the length of dataset
    def __len__(self):
        return len(self.meta)
