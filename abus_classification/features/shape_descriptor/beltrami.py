import trimesh
import numpy as np
import skimage
from scipy.sparse.linalg import eigs
from abus_classification.utils.mesh import get_mesh_from_3d_mask



def laplace_beltrami(mask_3d: np.ndarray) -> np.ndarray:
    mesh = get_mesh_from_3d_mask(mask_3d)
    # Calculate the Laplacian of the trimesh
    laplacian_sparse = trimesh.smoothing.laplacian_calculation(mesh, equal_weight=False)
    eigenvalues, eigenvectors = eigs(laplacian_sparse, k=6, which='SM')  # Get the smallest 6 eigenvalues and corresponding eigenvectors
    laplacian = trimesh.smoothing.laplacian_calculation(mesh, equal_weight=False)
    # Calculate the eigenvalues
    eigenvalues = np.linalg.eigvals(laplacian)
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors
