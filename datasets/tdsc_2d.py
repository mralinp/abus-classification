import os
import torch
from PIL import Image
import numpy as np
import random


train_list = None
validation_list = None


def split_data(m, b):

    global train_list, validation_list

    m_t = set(random.sample(m, int(0.8*len(m))))
    b_t = set(random.sample(b, int(0.8*len(b))))
    m = set(m)
    b = set(b)
    m_v = m-m_t
    b_v = b-b_t
    train_list = list(m_t) + list(b_t)
    validation_list = list(m_v) + list(b_v)
    random.shuffle(train_list)
    random.shuffle(validation_list)


class TDSC2D(torch.utils.data.Dataset):
    
    def __init__(self, path="./data/tdsc/slices", train=True, transforms=None):

        global train_list, validation_list
        self.path = path
        self.transforms = transforms
        malignant = os.listdir(f"{path}/data/0")
        benign = os.listdir(f"{path}/data/1")
        malignant = [(f"data/0/{item}", 0) for item in malignant]
        benign = [(f"data/1/{item}", 1) for item in benign]

        if not train_list or not validation_list:
            split_data(malignant, benign)

        if train:
            self.data_list = train_list
        else:
            self.data_list = validation_list
            
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

if __name__ == "__main__":
    dataset = TDSC2D(path="../data/tdsc/slices", train = True)
    print(len(dataset))
    sample = dataset[0]
        