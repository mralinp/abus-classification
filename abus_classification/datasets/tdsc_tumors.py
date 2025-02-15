import pandas as pd
import numpy as np

from abus_classification.datasets.tdsc import TDSC


class TDSCTumors(TDSC):
    def __init__(self, path="./data/tdsc", split=TDSC.DataSplits.TRAIN, transforms=None):
        super().__init__(path, split, transforms)
        self.bbx_metadata = pd.read_csv(f"{self.path}/{self.split}/bbx_labels.csv", dtype={
            'id': int, 
            'c_x': float, 
            'c_y': float, 
            'c_z': float, 
            'len_x': float,
            'len_y': float,
            'len_z': float,}, index_col='id')

        
    def __getitem__(self, index):
        x, m, y = super().__getitem__(index)
        c_x, c_y, c_z, len_x, len_y, len_z = self.bbx_metadata.iloc[index]

        z_s = int(c_z-len_z/2)
        z_e = int(c_z+len_z/2)

        y_s = int(c_y-len_y/2)
        y_e = int(c_y+len_y/2)

        x_s = int(c_x-len_x/2)
        x_e = int(c_x+len_x/2)

        # z, y, x
        x = x[z_s:z_e, y_s:y_e, x_s:x_e]
        m = m[z_s:z_e, y_s:y_e, x_s:x_e]
                
        # apply transformers if needed
        if self.transforms:
            for transform in self.transforms:
                x, m = transform(x,m)
                
        return x, m, y