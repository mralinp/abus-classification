import torch
import pandas as pd
import nrrd
import os


class TDSC(torch.utils.data.Dataset):

    def __init__(self, path_to_dataset: str = "../data/tdsc", type: str = "train") -> None:
        """
        TDSC dataset constructor,
        :param path_to_dataset: Root path to the dataset
        :param type: Which type of dataset to be created (train, test, validation)
        """

        self.path_to_dataset = f"{path_to_dataset}/{type}"
        
        if not os.path.exists(path_to_dataset):
            # download dataset
            pass
        self.meta = pd.read_csv(f"{self.path_to_dataset}/labels.csv", dtype={'Case_id': int, 'Label': str, 'Data_path': str, 'Mask_path': str}).set_index('case_id')
        
    def __getitem__(self, index) -> tuple:
        """
        Returns the related data point with as a tuple of (real volume, mask, label)
        :param index:
        :return:
        """

        data = self.meta.iloc[index]
        x, _ = nrrd.read(self.path_to_dataset + "/" + data.data_path.replace('\\', '/'))
        m, _ = nrrd.read(self.path_to_dataset + "/" + data.mask_path.replace('\\', '/'))
        return x, m, data.label

    def __len__(self) -> int:
        return len(self.meta)
