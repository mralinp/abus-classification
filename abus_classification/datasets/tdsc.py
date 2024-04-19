import os
import nrrd
import torch
import pandas as pd
from typing import Final

from .dataset import Dataset
from .google_drive_downloader import GoogleDriveDownloader


DATASET_URL: Final = "https://drive.google.com/file/d/1NsYIqatNp2D4yCj8PwZ8g9IbAuAdPX6F"


class TDSC(Dataset):

    def __init__(self, path_to_dataset: str = "./dataset/tdsc") -> None:
        
        super(TDSC, self).__init__(path_to_dataset)
        self.metadata = pd.read_csv(f"{self.path}/labels.csv", dtype={'Case_id': int, 'Label': str, 'Data_path': str, 'Mask_path': str}).set_index('case_id')
        
    def validate(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            drive_downloader = GoogleDriveDownloader(None)
            drive_downloader.download(DATASET_URL, f"{self.path}/dataset.zip")
            # Unzip the downloaded data
            drive_downloader.save()
        return True
        
        
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