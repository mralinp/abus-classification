import os
import json
import numpy as np
from abus_classification.datasets.dataset import Dataset
from abus_classification.datasets.tdsc_tumors import TDSCTumors
from abus_classification.utils.image import resample


class TDSCTumorsResampled(Dataset):
    
    def __init__(self, path="./data/tdsc/resample", transformers=None):
        super(TDSCTumorsResampled, self).__init__(path)
        self.transformers = transformers
        with open(f"{self.path}/labels.json", 'r') as meta_file:
            self.meta = json.load(meta_file).get('train')
        
    def validate(self):
        if not os.path.exists(f"{self.path}"):
            os.makedirs(self.path)
            os.makedirs(f"{self.path}/data")
            os.makedirs(f"{self.path}/mask")
            self.generate_data()
        return True
    
    def generate_data(self):
        print("Generating data....")
        train_meta = []
        tumors = TDSCTumors(f"{self.path}/..")
        for tumor_idx, (x,m,y) in enumerate(tumors):
            name = f"{tumor_idx}-{y}"
            x = resample(x)
            m = resample(m)
            np.save(f"{self.path}/data/{name}", x)
            np.save(f"{self.path}/mask/{name}", m)
            train_meta.append({"name": name, "label":y})        
        
        with open(f"{self.path}/labels.json", 'w') as f:
            json.dump({'train': train_meta}, f)
    
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
    