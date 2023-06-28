import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset.tdsc import TDSC
from tqdm import tqdm


dataset = TDSC(path_to_dataset="data/tdsc")

dx = {}
dy = {}
dz = {}

for datum in tqdm(dataset):
    x, y, z = datum[0].shape
    if x not in dx:
        dx[x] = 1
    else:
        dx[x] += 1

    if y not in dy:
        dy[y] = 1
    else:
        dy[y] += 1

    if z not in dz:
        dz[z] = 1
    else:
        dz[z] += 1

print(f"x: {dx}")
print(f"y: {dy}")
print(f"z: {dz}")

