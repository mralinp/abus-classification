import torch
import pandas as pd
import nrrd
import numpy as np
import cv2
import os
from . import utils
from . import static


training_data_list = [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 56, 58, 59, 61, 62, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 84, 85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
validation_data_list = [0, 3, 5, 15, 18, 30, 37, 38, 48, 55, 57, 60, 63, 66, 79, 82, 83, 86, 88, 89]

class TDSC(torch.utils.data.Dataset):

    def __init__(self, path_to_dataset: str = "./data/tdsc", train=True, transform=None, to_tensor=True) -> None:
        """
        TDSC dataset constructor,
        :param path_to_dataset: Root path to the dataset
        :param type: Which type of dataset to be created (train, test, validation)
        """

        global training_data_list, validation_data_list
        
        self.path_to_dataset = path_to_dataset
        self.metadata = pd.read_csv(f"{path_to_dataset}/labels.csv", dtype={'Case_id': int, 'Label': str, 'Data_path': str, 'Mask_path': str}).set_index('case_id')
        self.transform = transform
        self.to_tensor_output = to_tensor
        
        if train :
            self.data_list = training_data_list
        else:
            self.data_list = validation_data_list
        
    def __getitem__(self, index) -> tuple:
        """
        Returns the related data point with as a tuple of (real volume, mask, label)
        :param index:
        :return:
        """
        
        global tumor_locals

        label, vol_path, mask_path = self.metadata.iloc[self.data_list[index]]
        vol_path = vol_path.replace('\\', '/')
        mask_path = mask_path.replace('\\', '/')
        label = 0 if label == 'M' else 1
        
        
        vol, _ = nrrd.read(f"{self.path_to_dataset}/{vol_path}")
        mask, _ = nrrd.read(f"{self.path_to_dataset}/{mask_path}") 
        
        # # extracting tumors
        p1, p2 = static.tumor_locals[self.data_list[index]]
        vol = np.array(vol[p1[0]:p2[0], p1[1]:p2[1], p1[2]:p2[2]], dtype=np.float32)
        mask = np.array(mask[p1[0]:p2[0], p1[1]:p2[1], p1[2]:p2[2]], dtype=np.float32)
        
        vol, mask = utils.ResizeData((vol,mask), (128,128))
        
        # Apply transformers
        if self.transform is not None:
            for i in range(vol.shape[2]):
                augmentations = self.transform(image=vol[:,:,i], mask=mask[:,:,i])
                vol[:,:,i] = augmentations["image"]
                mask[:,:,i] = augmentations["mask"]
        
        return vol/255, mask, label

    def __len__(self) -> int:
        return len(self.data_list)
    
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = TDSC(path_to_dataset="./data/tdsc", train=False)
    print(len(dataset))
    vol,mask,label = dataset[1]
    print(vol.shape)
    print(mask.shape)
    
    x = vol[:,:,5]
    y = mask[:,:,5]
    
    plt.subplot(1,3,1)
    plt.imshow(x)
    plt.subplot(1,3,2)
    plt.imshow(y)
    x = x-x*y
    plt.subplot(1,3,3)
    plt.imshow(x)
    plt.show()
    
