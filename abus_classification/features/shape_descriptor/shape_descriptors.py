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

def lb_spectrum(mesh: trimesh.Trimesh, k: int = 50, scale_invariant: bool = True) -> np.ndarray:
    """
    First k Laplace-Beltrami eigenvalues of a closed mesh, sorted, the zero mode included.

    With scale_invariant, they are multiplied by the surface area. Scaling a shape by c
    divides lambda by c^2 and multiplies the area by c^2, so lambda * area ignores size.
    """
    L, M = compute_laplace_beltrami_operator(mesh)
    eigenvalues, _ = compute_eigendecomposition(L, M, k=k)
    return eigenvalues * mesh.area if scale_invariant else eigenvalues

def heat_trace_coefficients(eigenvalues: np.ndarray, area: float, n_times: int = 30,
                            tail: float = 8.0, tau_max: float = 0.02) -> Tuple[float, float, float]:
    """
    Fit Z(t) = sum exp(-lambda t) = A / (4 pi t) + chi / 6 + t / (60 pi) * int K^2 dA + ... (McKean and Singer 1967).

    Works in tau = t / area, which makes the three terms scale-free: Z = c0 / (4 pi tau) + chi / 6 + c2 tau, with c0 = 1
    on the true surface and c2 = area * int K^2 dA / (60 pi). Times start where the truncated tail is negligible,
    exp(-lambda_max t) = exp(-tail), so too few modes raise the first time, and tau_max keeps the expansion valid.

    Returns:
        (c0, chi, c2). Each eigenvalue list must come from one connected surface and include the zero mode.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float) * area
    tau = np.geomspace(tail / eigenvalues[-1], tau_max, n_times)
    trace = np.exp(-np.outer(tau, eigenvalues)).sum(axis=1)
    design = np.column_stack([1 / (4 * np.pi * tau), np.full(n_times, 1 / 6), tau])
    # Relative weights: Z spans orders of magnitude over the window.
    c0, chi, c2 = np.linalg.lstsq(design / trace[:, None], np.ones(n_times), rcond=None)[0]
    return float(c0), float(chi), float(c2)

def sphere_level_spread(eigenvalues: np.ndarray, levels: int = 6) -> np.ndarray:
    """
    How far each group of modes that coincide on a round sphere has split apart: std / mean of lambda within the group.

    On a sphere, degree l has 2l + 1 equal eigenvalues, at indices l^2 to l^2 + 2l. A small deformation by a spherical
    harmonic of even degree d splits level l at first order only if 2l >= d, so low levels ignore fine lobes.

    Args:
        eigenvalues: Sorted eigenvalues with the zero mode, at least (levels + 1)^2 of them.
        levels: Number of levels, starting at l = 1.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    groups = [eigenvalues[l ** 2:(l + 1) ** 2] for l in range(1, levels + 1)]
    return np.array([g.std() / g.mean() for g in groups])

def spectrum_bound(mesh: trimesh.Trimesh, deformed: trimesh.Trimesh) -> Tuple[float, float]:
    """
    Guaranteed range of lambda_k(deformed) / lambda_k(mesh), for every k, between two meshes with the same faces.

    On a triangle, a linear map with singular values s1 >= s2 scales the Dirichlet energy by at most s1 / s2 and the
    area by s1 * s2. Min-max over piecewise-linear functions then gives lower = 1 / (E * A_max), upper = E / A_min,
    with E the largest s1 / s2 and A the area factors. Only metric change counts, not how far points moved.
    """
    def frame(vertices):
        a = vertices[mesh.faces[:, 1]] - vertices[mesh.faces[:, 0]]
        b = vertices[mesh.faces[:, 2]] - vertices[mesh.faces[:, 0]]
        return np.stack([a, b], axis=-1)

    source, target = frame(mesh.vertices), frame(deformed.vertices)
    # The map's singular values are those of target * pinv(source), restricted to each triangle's plane.
    singular = np.linalg.svd(target @ np.linalg.pinv(source), compute_uv=False)[:, :2]
    conformal, area = (singular[:, 0] / singular[:, 1]).max(), singular[:, 0] * singular[:, 1]
    return float(1 / (conformal * area.max())), float(conformal / area.min())

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
