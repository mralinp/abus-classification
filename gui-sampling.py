import cv2
import numpy as np

import datasets


dataset = datasets.Tumors()
print(len(dataset))
center = (0,0)


def mouse_event(event, x,y, flags, params):
    global center
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        center = (y,x)

pad = 25

for idx, data in enumerate(dataset):
    print(f"sampling data {idx}/{len(dataset)}...")
    x,m,l = data
    _,_,depth = x.shape
    middle = depth//2
    samples = np.zeros((pad*2, pad*2, 3), dtype=np.uint8)
    for i in range(-1,2,1):
        sli = x[:,:,middle+i]
        sli_mask = m[:,:,middle+i]
        img = sli - sli_mask * sli * 0.5 
        img = img.astype(np.uint8)
        cv2.imshow('slice', img)
        cv2.setMouseCallback('slice', mouse_event)
        cv2.waitKey()
        cv2.destroyAllWindows()
        patch = sli[center[0]-pad:center[0]+pad,center[1]-pad:center[1]+pad]
        w,h = patch.shape
        samples[:w,:h,i+1] = patch
        cv2.imshow('res', samples[:,:,i+1])
        cv2.waitKey()
        cv2.destroyAllWindows()
    folder = 'malignant' if l == 0 else 'benign'
    np.save(f"./data/tdsc/texture/{folder}/{idx}", samples)
        
    