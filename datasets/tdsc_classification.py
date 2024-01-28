import os
import random
import torch
from PIL import Image
import numpy as np


class TDSCForClassification(torch.utils.data.Dataset):
    
    def __init__(self, path="./data/tdsc/classification", transforms=None):

        self.path = path
        self.transforms = transforms
        malignant = os.listdir(f"{path}/data/0")
        benign = os.listdir(f"{path}/data/1")
        malignant = [(f"data/0/{item}", 0) for item in malignant]
        benign = [(f"data/1/{item}", 1) for item in benign]
        print(f"#benign: {len(benign)}, #malignant: {len(malignant)}")
        self.data_list = malignant + benign
            
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_path, label = self.data_list[index]
        mask_path = data_path.replace('data', 'mask')
        image = np.array(Image.open(f"{self.path}/{data_path}").convert("L"), dtype=np.float32)
        mask = np.array(Image.open(f"{self.path}/{mask_path}").convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        if self.transforms is not None:
            augmentation = self.transforms(image=image,mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]
        return image, mask, float(label)
    
    def shuffle(self):
        random.shuffle(self.data_list)

if __name__ == "__main__":
    dataset = TDSCForClassification(path="../data/tdsc/classification")
    print(len(dataset))
    sample = dataset[0]
        