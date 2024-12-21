import numpy as np
from skimage import img_as_float


def obnlm(img, t, f, h):
    """
    Optimized Bayesian Non-Local Means filter
    Args:
        img: input image
        t: search window size
        f: similarity window size
        h: degree of filtering
    """
    img = img_as_float(img)
    m, n = img.shape
    img_denoised = np.zeros((m, n))
    h = h * h

    # Pre-compute normalized kernel for similarity window
    kernel = np.zeros((2*f+1, 2*f+1))
    for d in range(1, f+1):
        value = 1. / ((2*d+1) * (2*d+1))
        kernel[f-d:f+d+1, f-d:f+d+1] += value

    kernel = kernel / (f * np.sum(kernel))
    vkernel = kernel.reshape(-1)

    # Pad image for border handling
    pdimg = np.pad(img, f, mode='symmetric')

    # Denoising
    for i in range(m):
        for j in range(n):
            i1, j1 = i + f, j + f
            
            # Extract reference patch
            W1 = pdimg[i1-f:i1+f+1, j1-f:j1+f+1]
            
            # Define search window boundaries
            rmin = max(i1-t, f)
            rmax = min(i1+t, m+f-1)
            smin = max(j1-t, f)
            smax = min(j1+t, n+f-1)
            
            wmax = 0
            average = 0
            sweight = 0
            
            # Compare with neighboring patches
            for r in range(rmin, rmax+1):
                for s in range(smin, smax+1):
                    if (r == i1 and s == j1):
                        continue
                        
                    # Extract comparison patch
                    W2 = pdimg[r-f:r+f+1, s-f:s+f+1]
                    
                    # Calculate Pearson distance
                    temp = ((np.square(W1-W2))/W2).reshape(-1)
                    d = np.dot(vkernel, temp)
                    w = np.exp(-d/h)
                    
                    if w > wmax:
                        wmax = w
                    
                    sweight += w
                    average += w * pdimg[r, s]
            
            # Add contribution of central pixel
            average += wmax * pdimg[i1, j1]
            sweight += wmax
            
            # Compute final pixel value
            img_denoised[i, j] = average / sweight if sweight > 0 else img[i, j]
                
    return img_denoised
