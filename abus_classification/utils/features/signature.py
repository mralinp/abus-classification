import cv2
import numpy as np
from .. import image

def boundary_signature_2d(boundary_image: np, resolution:float=1., dif_min:int=5) -> np:
    
    assert dif_min >= 0
    assert resolution >= 0
    
    x0, y0 = image.find_shape_center(boundary_image)
    cords_x, cords_y = np.nonzero(boundary_image)
    boundary_cords = list(zip(cords_x, cords_y))
    signature = np.zeros([int(360/resolution)], dtype=np.float32)
    len_signature = signature.shape[0]

    for idx in range(len_signature):
        # m*x + c = y => c = -tan(theta)*x0 + y0
        theta = idx*resolution
        slop = np.tan(np.radians(theta))
        if theta == 90 or theta == 270:
            slop = 0
        c = -1*slop*x0+y0
        best_dif = 1000
        for x, y in boundary_cords:
            y_pred = slop*x + c
            dif = np.abs(y_pred - y)
            if dif < dif_min and dif < best_dif:
                best_dif = dif
                signature[idx] = np.sqrt((x0-x)**2 + (y0-y)**2)

    return signature


def boundary_signature_3d(binary_volume: np, resolution:tuple[float, float]=(1.,1.), dif_min:int=5)->np:
    
    assert dif_min > 0
    assert resolution[0] >= 0
    assert resolution[1] >= 0
    
    n_alpha_bins = 360 // resolution[0]
    n_alpha_bins = n_alpha_bins + 1 if 360%resolution[0] != 0 else n_alpha_bins
    n_beta_bins = 360//resolution[1]
    n_beta_bins = n_beta_bins + 1 if 360%resolution[1] != 0 else n_beta_bins
    
    n_alpha_bins, n_beta_bins = int(n_alpha_bins), int(n_beta_bins)
    sig_counter = [[0 for i in range(n_beta_bins)] for j in range(n_alpha_bins)]
    sig = [[0 for i in range(n_beta_bins)] for j in range(n_alpha_bins)]
    
    center = image.find_shape_center(binary_volume)
    boundary_points = image.find_surface_points_3d(binary_volume)
    
    for point in boundary_points:
        
        point_with_center_orig = point - center
        alpha = np.arctan(point_with_center_orig[1]/point_with_center_orig[0])
        beta = np.arctan(point_with_center_orig[2]/point_with_center_orig[0])
        
        alpha = np.degrees(alpha)
        beta = np.degrees(beta)
        
        alpha_bin = int(alpha//resolution[0])
        alpha_bin_diff = alpha_bin%resolution[0]
        beta_bin = int(beta//resolution[0])
        beta_bin_diff = beta%resolution[1]
        
        if alpha_bin_diff > resolution[0]/2:
            alpha_bin += 1
            alpha_bin_diff = resolution[0] - alpha_bin_diff
        
        if beta_bin_diff > resolution[1]/2:
            beta_bin += 1
            beta_bin_diff = resolution[1] - beta_bin_diff
                
        distance = np.sqrt((point**2).sum())
        sig[alpha_bin][beta_bin] += distance*(beta_bin_diff + alpha_bin_diff)
        sig_counter[alpha_bin][beta_bin] += 1
    
    for i in range(n_alpha_bins):
        for j in range(n_beta_bins):
            if sig_counter[i][j] > 0:
                sig[i][j] /= sig_counter[i][j]
            
    return np.array(sig, dtype=np.float32)