import cv2
import numpy as np
from abus_classification.utils import image


def boundary_signature3d(binary_volume: np, resolution:tuple[float, float]=(1.,1.), dif_min:int=5)->np:
    
    assert dif_min > 0
    assert resolution[0] >= 0
    assert resolution[1] >= 0
    
    n_alpha_bins = 360 // resolution[0]
    n_alpha_bins = n_alpha_bins + 1 if 360%resolution[0] != 0 else n_alpha_bins
    n_beta_bins = 360 // resolution[1]
    n_beta_bins = n_beta_bins + 1 if 360%resolution[1] != 0 else n_beta_bins
    
    n_alpha_bins, n_beta_bins = int(n_alpha_bins), int(n_beta_bins)
    
    sig_counter = np.zeros((n_alpha_bins, n_beta_bins), dtype=np.float32)
    sig = np.zeros((n_alpha_bins, n_beta_bins), dtype=np.int32)
    
    center = image.find_shape_center(binary_volume)
    boundary_points = image.get_surface_points(binary_volume)
    
    for point in boundary_points:
        
        point_with_center_orig = point - center
        alpha = np.arctan(point_with_center_orig[1]/point_with_center_orig[0])
        beta = np.arctan(point_with_center_orig[2]/point_with_center_orig[0])
        
        alpha = np.degrees(alpha)%180
        beta = np.degrees(beta)%180
        
                    
        alpha_bin = int(alpha//resolution[0])
        beta_bin = int(beta//resolution[0])
        
        alpha_bin_diff = alpha - alpha_bin*resolution[0]
        beta_bin_diff = beta - beta_bin*resolution[0]
        
        if alpha_bin_diff > 0.5:
            alpha_bin += 1
            alpha_bin_diff = resolution[0] - alpha_bin_diff
        
        if beta_bin_diff > 0.5:
            beta_bin += 1
            beta_bin_diff = resolution[1] - beta_bin_diff
            
        alpha_bin = alpha_bin%n_alpha_bins
        beta_bin = beta_bin%n_beta_bins
        
        distance = np.sqrt((point**2).sum())
        sig[alpha_bin,beta_bin] += distance*(beta_bin_diff + alpha_bin_diff)
        sig_counter[alpha_bin,beta_bin] += 1
    
    for i in range(n_alpha_bins):
        for j in range(n_beta_bins):
            if sig_counter[i][j] > 0:
                sig[i][j] /= sig_counter[i][j]
            
    return np.array(sig, dtype=np.float32)