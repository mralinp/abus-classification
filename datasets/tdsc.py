import torch
import pandas as pd
import nrrd
import numpy as np
import cv2
import os
import random


class TDSC(torch.utils.data.Dataset):

    def __init__(self, path_to_dataset: str = "./data/tdsc", train=True) -> None:
        """
        TDSC dataset constructor,
        :param path_to_dataset: Root path to the dataset
        :param type: Which type of dataset to be created (train, test, validation)
        """
        
        self.path_to_dataset = path_to_dataset
        self.metadata = pd.read_csv(f"{path_to_dataset}/labels.csv", dtype={'Case_id': int, 'Label': str, 'Data_path': str, 'Mask_path': str}).set_index('case_id')
        
    def __getitem__(self, index) -> tuple:
        """
        returns the related data point with as a tuple of (volume, mask, label)
        :param index:
        :return:
        """        
        label, vol_path, mask_path = self.metadata.iloc[index]
        vol_path = vol_path.replace('\\', '/')
        mask_path = mask_path.replace('\\', '/')
        label = 0 if label == 'M' else 1
        
        
        vol, _ = nrrd.read(f"{self.path_to_dataset}/{vol_path}")
        mask, _ = nrrd.read(f"{self.path_to_dataset}/{mask_path}") 
        
        return vol, mask, label

    def __len__(self) -> int:
        return len(self.metadata)
    

if __name__ == '__main__':
    dataset = TDSC(path_to_dataset="../data/tdsc", train=False)
    print(len(dataset))
    vol,mask,label = dataset[1]
    print(vol.shape)
    print(mask.shape)