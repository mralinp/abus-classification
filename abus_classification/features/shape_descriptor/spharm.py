"""Spherical harmonic description of closed genus-0 surfaces (SPHARM, Brechbühler et al. 1995), through a conformal sphere map."""
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.linalg import splu
from scipy.special import sph_harm_y

from abus_classification.features.shape_descriptor.shape_descriptors import compute_laplace_beltrami_operator


def balance(points: np.ndarray, weights: np.ndarray, steps: int = 50) -> np.ndarray:
    """Moebius maps of the sphere that put the weighted centre of mass at the origin, as in Hersch's proof."""
    weights = weights / weights.sum()
    for _ in range(steps):
        centre = weights @ points
        if np.linalg.norm(centre) < 1e-8:
            break
        # A damped step keeps the map inside the unit ball.
        c = 0.5 * centre
        shifted = points - c
        points = (1 - c @ c) * shifted / (shifted ** 2).sum(1, keepdims=True) - c
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


def sphere_map(mesh: trimesh.Trimesh, steps: int = 40, step_size: float = 0.05) -> np.ndarray:
    """
    Conformal map of a genus-0 surface onto the unit sphere by conformalized mean curvature flow (Kazhdan et al. 2012).

    The stiffness matrix stays that of the input, which keeps the flow from forming necks. Returns one unit vector per
    vertex, balanced by the input's vertex areas, so the map is fixed up to a rotation.
    """
    L, _ = compute_laplace_beltrami_operator(mesh)
    points = (mesh.vertices - mesh.vertices.mean(0)) / np.sqrt(mesh.area / (4 * np.pi))
    for _ in range(steps):
        current = trimesh.Trimesh(points, mesh.faces, process=False)
        _, M = compute_laplace_beltrami_operator(current)
        points = splu((M + step_size * L).tocsc()).solve(M @ points)
        points = points - (M.diagonal() @ points / M.diagonal().sum())
        points /= np.sqrt(trimesh.Trimesh(points, mesh.faces, process=False).area / (4 * np.pi))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    _, M = compute_laplace_beltrami_operator(mesh)
    return balance(points, M.diagonal())


def real_harmonics(points: np.ndarray, degree: int) -> np.ndarray:
    """Real orthonormal spherical harmonics at unit vectors, (n, (degree + 1)^2), ordered by l then m."""
    theta, phi = np.arccos(np.clip(points[:, 2], -1, 1)), np.arctan2(points[:, 1], points[:, 0])
    columns = []
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            y = sph_harm_y(l, abs(m), theta, phi)
            columns.append(y.real if m == 0 else np.sqrt(2) * (y.imag if m < 0 else y.real))
    return np.stack(columns, axis=1)


def spharm_coefficients(mesh: trimesh.Trimesh, sphere: np.ndarray, degree: int = 15, ridge: float = 1e-8) -> np.ndarray:
    """Coefficients of the surface's x, y, z over the sphere, ((degree + 1)^2, 3), by least squares on sphere areas."""
    _, M = compute_laplace_beltrami_operator(trimesh.Trimesh(sphere, mesh.faces, process=False))
    basis, weight = real_harmonics(sphere, degree), M.diagonal()[:, None]
    normal = basis.T @ (weight * basis) + ridge * np.eye(basis.shape[1])
    return np.linalg.solve(normal, basis.T @ (weight * mesh.vertices))


def spharm_power(coefficients: np.ndarray, degree: int = 15) -> np.ndarray:
    """Power per degree l = 0..degree, summed over m and x, y, z: unchanged by rotating the shape or the parametrisation."""
    return np.array([(coefficients[l ** 2:(l + 1) ** 2] ** 2).sum() for l in range(degree + 1)])


def map_quality(mesh: trimesh.Trimesh, sphere: np.ndarray) -> Tuple[float, float, float]:
    """(share of flipped triangles, median conformal stretch, largest area ratio over median on the sphere)."""
    normals = np.cross(sphere[mesh.faces[:, 1]] - sphere[mesh.faces[:, 0]], sphere[mesh.faces[:, 2]] - sphere[mesh.faces[:, 0]])
    flipped = (np.einsum("ij,ij->i", normals, sphere[mesh.faces].mean(1)) * np.sign(mesh.volume) < 0).mean()
    source = mesh.vertices[mesh.faces]; target = sphere[mesh.faces]
    a = np.stack([source[:, 1] - source[:, 0], source[:, 2] - source[:, 0]], -1)
    b = np.stack([target[:, 1] - target[:, 0], target[:, 2] - target[:, 0]], -1)
    s = np.linalg.svd(b @ np.linalg.pinv(a), compute_uv=False)[:, :2]
    ratio = s[:, 0] * s[:, 1]
    return float(flipped), float(np.median(s[:, 0] / s[:, 1])), float(ratio.max() / np.median(ratio))
