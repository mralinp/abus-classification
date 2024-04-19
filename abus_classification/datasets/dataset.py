from abc import ABC, abstractmethod
import torch


class Dataset(ABC, torch.utils.data.Dataset):
    
    def __init__(self, path:str):
        self.path = path
        if not self.validate():
            raise FileNotFoundError("data files are corrupted.")
    
    @abstractmethod
    def validate(self):
        pass
    
    
        