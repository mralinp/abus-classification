import multiprocessing
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import sys
# sys.path.append("../")
import abus_classification


dataset = abus_classification.datasets.TDSCTumors(path="./data/tdsc")    


def extract_signature(data):
    x, y = data
    sig = abus_classification.utils.features.boundary_signature_3d(x, resolution=(5,5))
    return sig, y

signature_dataset = []
num_processes = multiprocessing.cpu_count()

loop = tqdm(range(0, 100, num_processes))

for i in loop:
    loop.set_postfix(processing=f"{i} to {i+num_processes}")
    data_chunk = []
    for j in range(i, i+num_processes):
        _, x, y = dataset[j]
        data_chunk.append((x,y))
    with multiprocessing.Pool(processes=num_processes) as pool:
        signature_dataset.append(pool.map(extract_signature, data_chunk))
    