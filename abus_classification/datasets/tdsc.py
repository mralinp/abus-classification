import os
import nrrd
import torch
import pandas as pd
from typing import Final

from .dataset import Dataset
from .google_drive_downloader import GoogleDriveDownloader


DATASET_IDS: Final = {
    "data_0": "1K762mw8vAIRjgoeR7aJN-0NIntOBUb6O",
    "data_1": "17Umzl10lpFu4mGJ9HrZrC-Lcc-klKcl1",
    "mask": "1Z2RUoUoOukA93LyTgwKPKCrfFePva1pV",
    "labels": "1Fn6psOjknovxmShESRYpaxDAbb9txvH7",
    "bbx_labels": "1firgUGMMMscXoYlQCzdg8Y7x2Hc3enIt"
}


class TDSC(Dataset):

    def __init__(self, path_to_dataset: str = "./data/tdsc") -> None:
        
        super(TDSC, self).__init__(path_to_dataset)
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
        downloader.download(DATASET_IDS.get("data_0"), f"{self.path}/DATA/data0.zip")
        downloader.save()
        downloader.download(DATASET_IDS.get("data_1"), f"{self.path}/DATA/data1.zip")
        downloader.save()
        downloader.download(DATASET_IDS.get("mask"), f"{self.path}/MASK/mask.zip")
        downloader.save()
        downloader.download(DATASET_IDS.get("labels"), f"{self.path}/labels.csv")
        downloader.download(DATASET_IDS.get("bbx_labels"), f"{self.path}/bbx_labels.csv")
        
        
        
    def __getitem__(self, index) -> tuple:
        label, vol_path, mask_path = self.metadata.iloc[index]
        vol_path = vol_path.replace('\\', '/')
        mask_path = mask_path.replace('\\', '/')
        label = 0 if label == 'M' else 1
        
        
        vol, _ = nrrd.read(f"{self.path}/{vol_path}")
        mask, _ = nrrd.read(f"{self.path}/{mask_path}") 
        
        return vol, mask, label

    def __len__(self) -> int:
        return len(self.metadata)