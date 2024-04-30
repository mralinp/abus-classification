import os
import json
import numpy as np
from .dataset import Dataset
from .tdsc_tumors import TDSCTumors
from sklearn.model_selection import train_test_split


class TDSCTumors2D(Dataset):
    
    
    def __init__(self, path="./data/tdsc/2d", train=True, transformers=None):
        super(TDSCTumors2D, self).__init__(path)
        self.transformers = transformers
        if train:
            with open(f"{self.path}/labels.json", 'r') as meta_file:
                self.meta = json.load(meta_file).get('train')
        else:
            with open(f"{self.path}/labels.json", 'r') as meta_file:
                self.meta = json.load(meta_file).get('test')
    
    
    def validate(self):
        if not os.path.exists(f"{self.path}"):
            os.makedirs(self.path)
            os.makedirs(f"{self.path}/data")
            os.makedirs(f"{self.path}/mask")
            self.generate_data()
        
        return True
        
    
    def generate_data(self):
        print("Generating data....")
        malignant = []
        benign = []
        tumors = TDSCTumors(f"{self.path}/..")
        for tumor_idx, (x,m,y) in enumerate(tumors):
            x = np.transpose(x, (2,0,1))
            m = np.transpose(m, (2,0,1))
            for i in range(len(x)):
                data = x[i]
                mask = m[i]
                name = f"{tumor_idx}-{i}-{y}"
                np.save(f"{self.path}/data/{name}", data)
                np.save(f"{self.path}/mask/{name}", mask)
                if y == 0:
                    malignant.append({"name": name, "label":y})
                else:
                    benign.append({"name": name, "label":y})
                    
        train_m, test_m = train_test_split(malignant, test_size=.2, random_state=42)
        train_b, test_b = train_test_split(benign, test_size=.2, random_state=42)
        train_meta = train_m + train_b
        test_meta = test_m + test_b
        
        with open(f"{self.path}/labels.json", 'w') as f:
            json.dump({'train': train_meta, 'test': test_meta}, f)
    
    
    def __len__(self):
        return len(self.meta)
    
    
    def __getitem__(self, index):
        meta_data = self.meta[index]
        name = meta_data.get('name')
        label = meta_data.get('label')
        mask = np.load(f"{self.path}/mask/{name}.npy")
        data = np.load(f"{self.path}/data/{name}.npy")
        
        if self.transformers:
            for transformer in self.transformers:
                data, mask = transformer(data, mask)

        return data, mask, label