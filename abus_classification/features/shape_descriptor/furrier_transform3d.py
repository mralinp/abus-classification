import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes

def extract_boundary(mask):
    """
    Extract the surface boundary of a 3D object using the Marching Cubes algorithm.
    Args:
        mask (np.ndarray): 3D binary mask.
    Returns:
        tuple: Vertices and faces of the surface mesh.
    """
    vertices, faces, _, _ = marching_cubes(mask, level=0.5)
    return vertices, faces

def compute_3d_fft(mask, vertices):
    """
    Compute the 3D FFT of the boundary points.
    Args:
        mask (np.ndarray): 3D binary mask (used to determine grid size).
        vertices (np.ndarray): Boundary points as extracted by Marching Cubes.
    Returns:
        np.ndarray: FFT result.
    """
    signal_3d = np.zeros_like(mask, dtype=np.complex128)
    for vx, vy, vz in vertices.astype(int):
        signal_3d[vx, vy, vz] = 1 + 0j  # Assign 1 at the boundary points
    fft_result = np.fft.fftn(signal_3d)
    return fft_result


def fft3d_signature(tumor_mask):
    # Step 1: Extract the boundary
    vertices, _ = extract_boundary(tumor_mask)
    # Step 2: Compute the 3D FFT
    fft_result = compute_3d_fft(tumor_mask, vertices)
    return fft_result
