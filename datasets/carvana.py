import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Carvana(Dataset):
    
    def __init__(self, path_to_dataset: str = "./data/carvana/train" , transform=None):
        self.data_path = path_to_dataset
        self.transform = transform
        self.images = os.listdir(f"{path_to_dataset}/data")        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(f'{self.data_path}/data', self.images[index])
        mask_path = os.path.join(f'{self.data_path}/mask', self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
    
    
if __name__ == "__main__":
    dataset = Carvana()
    print (len(dataset))
