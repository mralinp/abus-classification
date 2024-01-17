import os
import torch
from PIL import Image
import numpy as np


class TDSCForClassificationWithSub(torch.utils.data.Dataset):
    
    def __init__(self, path="./data/tdsc/classification_with_subsampling", transforms=None):

        self.path = path
        self.transforms = transforms
        malignant = os.listdir(f"{path}/0")
        benign = os.listdir(f"{path}/1")
        malignant = [(f"0/{item}", 0) for item in malignant]
        benign = [(f"1/{item}", 1) for item in benign]
        self.data_list = malignant + benign
            
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_path, label = self.data_list[index]
        image = np.array(Image.open(f"{self.path}/{data_path}").convert("L"), dtype=np.float32)
        if self.transforms is not None:
            augmentation = self.transforms(image=image)
            image = augmentation["image"]
        return image, float(label)


if __name__ == "__main__":
    dataset = TDSCForClassificationWithSub(path="../data/tdsc/slices")
    print(len(dataset))
    sample = dataset[0]
        