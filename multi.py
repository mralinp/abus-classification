import multiprocessing
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


import abus_classification


dataset = abus_classification.datasets.TDSCTumors(path="./data/tdsc")    

def extract_signature(data):
    signature = []
    x,y = data
    sig = abus_classification.utils.features.boundary_signature_3d(x, resolution=(5,5))
    signature.append((sig.flatten(), y))
    return signature

signature_dataset = []
num_processes = multiprocessing.cpu_count() - 2
chunk_size = len(dataset)//num_processes

for i in range(0, len(dataset), 10):
    data_chunks = []
    for j in range(i, i+chunk_size):
        _,x,y = dataset[j]
        data_chunks.append((x,y))
    print(f"Processing {i} to {j}")
    pool = multiprocessing.Pool(processes=num_processes)
    signature_dataset.append(pool.map(extract_signature, data_chunks))
    pool.close()
    pool.join()


X, Y = zip(*signature_dataset)
X = list(X)
Y = list(Y)

def classify_with_svm(x, y):
    
    acc = 0
    train_acc = 0
    cfm = [[0,0],
           [0,0]]

    for i in tqdm(range(100)):
        x_test, y_test = x[i], y[i]
        X_train, Y_train = x[:i] + x[i+1:], y[:i] + y[i+1:]
        clf = SVC()
        clf.fit(X_train, Y_train)
        P = clf.predict(X_train)
        train_acc += accuracy_score(Y_train, P)
        res = clf.predict([x_test])[0]
        if res == y_test:
            acc += 1
            
        cfm[y_test][res] += 1
            
    print(f"Train accuracy: {train_acc/100}")
    print(f"Accuracy: {acc/100}")
    print(f"{cfm[0]}\n{cfm[1]}")
    
classify_with_svm(X,Y)

def normalize_(x:np, interval=(0,1)):
    x_std = (x - x.min())/x.max()-x.min()
    mi, ma = interval
    return x_std*(ma-mi) + mi

def normalize(x):
    for i in range(100):
        x[i] = normalize_(x[i])
    return x

X = normalize(X)
classify_with_svm(X,Y)