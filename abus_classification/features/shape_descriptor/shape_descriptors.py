import numpy as np
import trimesh
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from typing import Tuple, Optional

def compute_laplace_beltrami_operator(mesh: trimesh.Trimesh) -> Tuple[csr_matrix, csr_matrix]:
    """
    Compute the Laplace-Beltrami operator using the cotangent weights scheme.
    Returns both the Laplacian matrix L and the mass matrix M.
    """
    vertices = mesh.vertices
    faces = mesh.faces
    n_vertices = len(vertices)
    
    # Initialize the matrices
    L = sp.lil_matrix((n_vertices, n_vertices))
    M = sp.lil_matrix((n_vertices, n_vertices))
    
    # Compute cotangent weights
    for face in faces:
        # Get vertices of the face
        vi = vertices[face]
        # Compute edges
        edges = np.roll(vi, -1, axis=0) - vi
        # Compute squared lengths of edges
        sq_lengths = np.sum(edges**2, axis=1)
        # Compute angles using cosine law
        angles = []
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            cos_angle = np.dot(edges[j], -edges[k]) / np.sqrt(sq_lengths[j] * sq_lengths[k])
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(angle)
        angles = np.array(angles)
        
        # Compute cotangent weights
        cotangents = 1.0 / np.tan(angles)
        
        # Fill the Laplacian matrix
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            L[face[i], face[j]] += cotangents[k] / 2
            L[face[j], face[i]] += cotangents[k] / 2
    
    # Make Laplacian symmetric
    L = L.tocsr()
    L = (L + L.T) / 2
    
    # Compute diagonal elements
    L.setdiag(-np.array(L.sum(axis=1)).flatten())
    
    # Compute mass matrix (area weights)
    for face in faces:
        # Compute area of the face
        vi = vertices[face]
        area = np.linalg.norm(np.cross(vi[1] - vi[0], vi[2] - vi[0])) / 2
        # Add area contribution to each vertex
        for v in face:
            M[v, v] += area / 3
    
    return L.tocsr(), M.tocsr()

def compute_eigendecomposition(L: csr_matrix, M: csr_matrix, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the first k eigenvalues and eigenvectors of the generalized eigenvalue problem L φ = λ M φ
    """
    eigenvalues, eigenvectors = eigs(L, k=k, M=M, which='SM')
    # Sort by eigenvalues
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real
    return eigenvalues, eigenvectors

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
