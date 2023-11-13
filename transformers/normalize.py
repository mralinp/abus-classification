import numpy as np


class Normalizer:
    def __call__(self, sample):
        x, m = sample
        return x/255, m/np.max(m)