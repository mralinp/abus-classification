import os
import nrrd
import pandas as pd
from typing import Final

from .dataset import Dataset
from .google_drive_downloader import GoogleDriveDownloader


DATASET_IDS: Final = {
    "data_0.zip": {
        "id":"1K762mw8vAIRjgoeR7aJN-0NIntOBUb6O",
        "path": "DATA",
        "zip": True
    },
    "data_1.zip": {
        "id":"17Umzl10lpFu4mGJ9HrZrC-Lcc-klKcl1",
        "path": "DATA",
        "zip": True
    },
    "mask.zip": {
        "id":"1Z2RUoUoOukA93LyTgwKPKCrfFePva1pV",
        "path": "MASK",
        "zip": True
    },
    "labels.csv": {
        "id":"1Fn6psOjknovxmShESRYpaxDAbb9txvH7",
        "path": "."
    },
    "bbx_labels.csv": {
        "id":"1firgUGMMMscXoYlQCzdg8Y7x2Hc3enIt",
        "path": "."
    }
}


class TDSC(Dataset):

    def __init__(self, path_to_dataset: str = "./data/tdsc", transforms=None):
        super(TDSC, self).__init__(path_to_dataset)
        self.do_transform = True
        self.transforms = transforms
        self.metadata = pd.read_csv(f"{self.path}/labels.csv", dtype={'Case_id': int, 'Label': str, 'Data_path': str, 'Mask_path': str}).set_index('case_id')
        
    def validate(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            os.makedirs(f"{self.path}/DATA")
            os.makedirs(f"{self.path}/MASK")
            self.download_data()
        return True
    
    def download_data(self):
        downloader = GoogleDriveDownloader(None)
        for file_name, file_info in DATASET_IDS.items:
            downloader.download(file_info.get("id"), f"{self.path}/{file_info.get("path")}/{file_name}")
            if file_info.get("zip"):
                downloader.save()
        
    def __getitem__(self, index) -> tuple:
        label, vol_path, mask_path = self.metadata.iloc[index]
        vol_path = vol_path.replace('\\', '/')
        mask_path = mask_path.replace('\\', '/')
        label = 0 if label == 'M' else 1
        
        
        vol, _ = nrrd.read(f"{self.path}/{vol_path}")
        mask, _ = nrrd.read(f"{self.path}/{mask_path}") 
        
        if self.transforms and self.do_transform:
            for transformer in self.transforms:
                vol, mask = transformer(vol, mask)
        
        return vol, mask, label

    def __len__(self) -> int:
        return len(self.metadata)