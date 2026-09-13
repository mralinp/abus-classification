import numpy as np
import trimesh
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from typing import Tuple, Optional

def compute_laplace_beltrami_operator(mesh: trimesh.Trimesh) -> Tuple[csr_matrix, csr_matrix]:
    """
    Compute the Laplace-Beltrami operator using the cotangent weights scheme.
    Returns both the Laplacian matrix L and the lumped mass matrix M.

    L is positive semi-definite (L = D - W), so the generalised eigenvalues of
    L phi = lambda M phi are >= 0 with lambda_0 = 0. The heat, wave and global
    point signatures below all assume that convention.

    The edge weight is w_ij = (cot a_ij + cot b_ij) / 2, where a_ij and b_ij
    are the angles opposite edge (i, j) in its two adjacent triangles.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    n_vertices = len(vertices)

    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]

    def cot(a, b):
        cross = np.linalg.norm(np.cross(a, b), axis=1)
        return np.einsum("ij,ij->i", a, b) / np.maximum(cross, 1e-12)

    # Angle at each corner, paired with the edge opposite it.
    cot0 = cot(v1 - v0, v2 - v0)   # opposite edge (1, 2)
    cot1 = cot(v2 - v1, v0 - v1)   # opposite edge (2, 0)
    cot2 = cot(v0 - v2, v1 - v2)   # opposite edge (0, 1)

    rows = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])
    weights = 0.5 * np.concatenate([cot0, cot1, cot2])

    W = sp.coo_matrix((weights, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
    W = W + W.T
    L = sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W

    # Lumped mass matrix: a third of each adjacent triangle's area per vertex.
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    mass = np.bincount(faces.ravel(), weights=np.repeat(areas / 3, 3), minlength=n_vertices)
    M = sp.diags(mass)

    return L.tocsr(), M.tocsr()

def compute_eigendecomposition(L: csr_matrix, M: csr_matrix, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the first k eigenvalues and eigenvectors of the generalized eigenvalue problem L φ = λ M φ

    Uses the symmetric solver in shift-invert mode around a small negative
    shift, which finds the smallest eigenvalues reliably even though L is
    singular.
    """
    eigenvalues, eigenvectors = eigsh(L, k=k, M=M, sigma=-1e-8, which="LM")
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]

def compute_hks(eigenvalues: np.ndarray, eigenvectors: np.ndarray, time_points: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the Heat Kernel Signature (HKS)
    """
    if time_points is None:
        time_points = np.logspace(-2, 2, 100)
    
    hks = np.zeros((eigenvectors.shape[0], len(time_points)))
    for i, t in enumerate(time_points):
        weights = np.exp(-eigenvalues * t)
        hks[:, i] = np.sum(weights * eigenvectors**2, axis=1)
    return hks

def compute_wks(eigenvalues: np.ndarray, eigenvectors: np.ndarray, energy_points: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the Wave Kernel Signature (WKS)
    """
    if energy_points is None:
        log_eigenvalues = np.log(np.maximum(eigenvalues, 1e-6))
        energy_points = np.linspace(log_eigenvalues[1], log_eigenvalues[-1], 100)
    
    sigma = (energy_points[1] - energy_points[0]) * 7
    wks = np.zeros((eigenvectors.shape[0], len(energy_points)))
    
    for i, e in enumerate(energy_points):
        weights = np.exp(-(e - np.log(np.maximum(eigenvalues, 1e-6)))**2 / (2 * sigma**2))
        weights /= np.sum(weights)
        wks[:, i] = np.sum(weights * eigenvectors**2, axis=1)
    
    return wks

def compute_gps(eigenvalues: np.ndarray, eigenvectors: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Compute the Global Point Signature (GPS)
    Uses the first k eigenvectors scaled by the inverse square root of their eigenvalues
    """
    # Avoid division by zero for the first eigenvalue (which should be approximately zero)
    scaling = 1.0 / np.sqrt(np.maximum(eigenvalues[1:k+1], 1e-10))
    return eigenvectors[:, 1:k+1] * scaling

def compute_agps(eigenvalues: np.ndarray, eigenvectors: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Compute the Augmented Global Point Signature (AGPS)
    Combines GPS with additional geometric features
    """
    # Compute regular GPS
    gps = compute_gps(eigenvalues, eigenvectors, k)
    
    # Add eigenvalue-weighted coordinates
    agps = np.zeros((eigenvectors.shape[0], k * 2))
    agps[:, :k] = gps  # Regular GPS features
    
    # Add weighted eigenvector features
    for i in range(k):
        agps[:, k+i] = eigenvectors[:, i+1] * np.sqrt(eigenvalues[i+1])
    
    return agps
