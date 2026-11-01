"""
Surface descriptors designed from the clinical signs of malignancy, using the Laplace-Beltrami operator.

The eigenvalues of a surface ignore how it bends in space. The operator applied to the surface's own coordinates does not,
because Delta x = -2 H n, with H the mean curvature (Meyer et al., 2003). Expanding the coordinates in the eigenfunctions,
c_k = <x, phi_k>, makes three classical quantities moments of one spectrum:

    M_0 = sum |c_k|^2            = integral |x - centre|^2 dA      how far the surface lies from its centre
    M_1 = sum lam_k |c_k|^2      = 2 area                          exact for the cotangent Laplacian
    M_2 = sum lam_k^2 |c_k|^2    = 4 W, W = integral H^2 dA        bending, the Willmore energy, at least 4 pi

The sums run over k >= 1. M_1 makes lam_k |c_k|^2 / (2 area) a budget: the share of the surface's area spent at each frequency.
A round sphere spends all of it at one frequency. By Cauchy-Schwarz, M_1^2 <= M_0 M_2, with equality only if
Delta x = -mu (x - centre), which by Takahashi's theorem (1966) makes a closed surface in R^3 a round sphere.

Three signs need more than the surface. A lesion's tilt to the skin needs the ultrasound beam, which rotation invariance
discards. Spicules, seen in the tissue around a lesion, need the image: `radial_lines` asks whether lines in coronal planes
run out of the lesion's surface or wrap around it, in the spirit of the collagen signatures of Provenzano et al. (2006).

References:
    M. Meyer, M. Desbrun, P. Schroeder, A. H. Barr. Discrete differential-geometry operators for triangulated 2-manifolds.
    Visualization and Mathematics III, 35-57, 2003.

    T. Takahashi. Minimal immersions of Riemannian manifolds. J. Math. Soc. Japan 18(4):380-385, 1966.

    D. Cohen-Steiner, H. Edelsbrunner, J. Harer. Stability of persistence diagrams. Discrete Comput. Geom. 37:103-120, 2007.

    P. P. Provenzano et al. Collagen reorganization at the tumor-stromal interface facilitates local invasion.
    BMC Medicine 4:38, 2006.
"""
import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from abus_classification.features.shape_descriptor.shape_descriptors import (
    compute_eigendecomposition, compute_laplace_beltrami_operator, weighted_quantiles)
from abus_classification.features.tan.spiculation import line_orientation

BODY_MODES = 16                                      # the zero mode and degrees 1 to 3 on a sphere: 1 + 3 + 5 + 7
LOBE_PROMINENCE = 0.05                               # a lobe rises this far above its saddle, as a fraction of the radius
SHELL_MM, SHELLS_MM = (2.0, 10.0), (2.0, 4.0, 6.0, 8.0, 10.0)   # from past the lesion's own edge response to about Tan's first r_max
FIELD_TAU = 0.002                                    # heat time of the line field, as t / area: about a third of the radius


def outward_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """Area-weighted unit vertex normals, pointing out of a closed surface."""
    normals = np.zeros_like(mesh.vertices)
    np.add.at(normals, mesh.faces.ravel(), np.repeat(mesh.face_normals * mesh.area_faces[:, None], 3, axis=0))
    return np.sign(mesh.volume) * normals / np.linalg.norm(normals, axis=1, keepdims=True)


def mesh_spectrum(mesh: trimesh.Trimesh, k: int = 200) -> dict:
    """Eigenvalues lam, eigenfunctions phi, vertex areas mass and coordinate spectrum c, of shape (k, 3), of a closed surface."""
    L, M = compute_laplace_beltrami_operator(mesh)
    lam, phi = compute_eigendecomposition(L, M, k=min(k, len(mesh.vertices) - 2))
    mass = M.diagonal()
    return dict(lam=lam, phi=phi, mass=mass, c=phi.T @ (mass[:, None] * mesh.vertices))


def spectral_moments(lam: np.ndarray, c: np.ndarray) -> list:
    """M_0, M_1 and M_2 of the coordinate spectrum, without the zero mode."""
    power = (c[1:] ** 2).sum(axis=1)
    return [float((lam[1:] ** p * power).sum()) for p in range(3)]


def spectral_roundness(lam: np.ndarray, c: np.ndarray) -> float:
    """M_1^2 / (M_0 M_2), in (0, 1]; 1 only for a round sphere. It ignores position, rotation and scale, but not bending."""
    m0, m1, m2 = spectral_moments(lam, c)
    return m1 ** 2 / (m0 * m2)


def area_budget(lam: np.ndarray, c: np.ndarray, area: float, degrees: int = 13) -> np.ndarray:
    """Share of the surface's area spent at degrees 1 to `degrees`, taking modes in the sphere's groups: k in [l^2, (l + 1)^2)."""
    spent = lam[1:] * (c[1:] ** 2).sum(axis=1) / (2 * area)
    return np.array([spent[l ** 2 - 1:(l + 1) ** 2 - 1].sum() for l in range(1, degrees + 1)])


def surface_height(mesh: trimesh.Trimesh, phi: np.ndarray, c: np.ndarray, body_modes: int = BODY_MODES) -> np.ndarray:
    """Signed height of each vertex above the surface's own smooth body, the part its low modes rebuild."""
    return ((mesh.vertices - phi[:, :body_modes] @ c[:body_modes]) * outward_normals(mesh)).sum(axis=1)


def mean_curvature(lam: np.ndarray, phi: np.ndarray, c: np.ndarray) -> np.ndarray:
    """|H| at each vertex from the modes kept, since Delta x = -2 H n."""
    return 0.5 * np.linalg.norm(phi[:, 1:] @ (lam[1:, None] * c[1:]), axis=1)


def surface_peaks(values: np.ndarray, mesh: trimesh.Trimesh) -> tuple:
    """
    Peak vertices of a function on a mesh and their prominences, the 0-dimensional persistence of its superlevel sets.

    Sweeping down from the top, a peak is born at a local maximum and dies where its region meets a higher peak's; its
    prominence is the drop between. A change of the function by at most e moves these by at most 2 e and can only add or
    remove peaks of prominence up to 2 e (Cohen-Steiner et al., 2007), unlike a plain count of maxima. The highest peak is
    given the whole range of the function.
    """
    parent = np.full(len(values), -1)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    neighbours, peaks = mesh.vertex_neighbors, {}
    order = np.argsort(-values, kind="stable")
    for v in order:
        roots = {root(n) for n in neighbours[v] if parent[n] >= 0}
        if not roots:
            parent[v] = v
            continue
        top = max(roots, key=lambda r: values[r])
        parent[v] = top
        for r in roots - {top}:
            parent[r] = top
            peaks[r] = values[r] - values[v]
    peaks[order[0]] = values.max() - values.min()
    vertices = np.fromiter(peaks, dtype=int)
    return vertices, np.array([peaks[v] for v in vertices])


def beam_orientation(c: np.ndarray, depth_axis: int, body_modes: int = BODY_MODES) -> tuple:
    """
    How a surface's smooth body sits relative to the beam, from how its low modes embed in space.

    Returns the share of the body's spread that lies along the beam (a third for a round shape, more when taller than wide)
    and the angle in degrees between its longest axis and the plane parallel to the skin (0 when parallel).
    """
    q = c[1:body_modes].T @ c[1:body_modes]
    axes = np.linalg.eigh(q)[1]
    return float(q[depth_axis, depth_axis] / np.trace(q)), float(np.degrees(np.arcsin(min(abs(axes[depth_axis, -1]), 1.0))))


def surface_signs(mesh: trimesh.Trimesh, spectrum: dict, depth_axis: int) -> dict:
    """The scale-free surface descriptors of one closed mesh. Height and curvature are relative to the area radius."""
    lam, phi, mass, c = spectrum["lam"], spectrum["phi"], spectrum["mass"], spectrum["c"]
    radius = np.sqrt(mesh.area / (4 * np.pi))
    height = surface_height(mesh, phi, c) / radius
    _, prominence = surface_peaks(height, mesh)
    prominence = np.sort(prominence)[::-1][1:]                                        # without the whole range the highest peak was given
    depth_share, tilt = beam_orientation(c, depth_axis)
    return dict(roundness=spectral_roundness(lam, c), bending=spectral_moments(lam, c)[2] / (16 * np.pi), budget=area_budget(lam, c, mesh.area),
                lobes=int((prominence >= LOBE_PROMINENCE).sum()) + 1, prominence=float(prominence.sum()), top_prominence=np.pad(prominence, (0, 40))[:40],
                roughness=float(np.sqrt(mass @ height ** 2 / mass.sum())), depth_share=depth_share, tilt=tilt,
                angularity=float(weighted_quantiles(mean_curvature(lam, phi, c) * radius, mass, np.array([0.99]))[0]))


def line_field(volume: np.ndarray, spacing, depth_axis: int, sigma_mm: float = 0.9) -> tuple:
    """Line direction and strength in every coronal plane, with Tan's detector. Arrays are (planes, rows, cols), plane 0 at the skin."""
    pixel = float(np.mean(np.delete(np.asarray(spacing), depth_axis)))
    return line_orientation(np.moveaxis(volume, depth_axis, 0), [sigma_mm / pixel])


def radial_lines(theta: np.ndarray, strength: np.ndarray, mask: np.ndarray, spacing, depth_axis: int, shell_mm=SHELL_MM) -> np.ndarray:
    """
    For every pixel around a lesion, whether its line runs out of the lesion or along it, in coronal planes through the lesion.

    The excess is |cos(line direction - direction from the nearest lesion pixel)| minus its expectation for a line drawn from
    the orientations of these planes, so that the anisotropy of ultrasound speckle cancels. It is positive for lines running
    out of the lesion and negative for lines wrapping around it, and the inner radius skips the lesion's own edge response.

    Args:
        theta, strength: From :func:`line_field`.
        mask: Lesion mask on the volume's own axes.
        spacing: Voxel size in mm along each axis.
        depth_axis: Axis along the beam.
        shell_mm: Inner and outer distance from the lesion.

    Returns:
        Array with one row per pixel: plane, row and column of the nearest lesion pixel, distance in mm, excess, line strength.
    """
    m = np.moveaxis(mask, depth_axis, 0)
    pixel = np.delete(np.asarray(spacing, dtype=float), depth_axis)
    planes = np.flatnonzero(m.any(axis=(1, 2)))
    bins = (np.arange(36) + 0.5) * np.pi / 36
    weights = np.histogram(theta[planes][~m[planes]], bins=36, range=(0, np.pi), weights=strength[planes][~m[planes]])[0]
    weights = weights / weights.sum()
    rows = []
    for k in planes:
        distance, (near_r, near_c) = ndimage.distance_transform_edt(~m[k], sampling=pixel, return_indices=True)
        r, c = np.nonzero((distance >= shell_mm[0]) & (distance < shell_mm[1]))
        direction = np.arctan2((c - near_c[r, c]) * pixel[1], (r - near_r[r, c]) * pixel[0])
        chance = np.abs(np.cos(bins[None] - direction[:, None])) @ weights
        rows.append(np.c_[np.full(len(r), k), near_r[r, c], near_c[r, c], distance[r, c],
                          np.abs(np.cos(theta[k, r, c] - direction)) - chance, strength[k, r, c]])
    return np.vstack(rows)


def line_summary(lines: np.ndarray, centre_plane: float, shells_mm=SHELLS_MM) -> dict:
    """Strength-weighted mean excess overall, on the skin side of the lesion's centre, and by distance from the lesion."""
    def mean(keep):
        return float(np.average(lines[keep, 4], weights=lines[keep, 5])) if lines[keep, 5].sum() > 0 else np.nan
    return dict(radial=mean(np.ones(len(lines), dtype=bool)), radial_upper=mean(lines[:, 0] <= centre_plane),
                radial_by_shell=np.array([mean((lines[:, 3] >= a) & (lines[:, 3] < b)) for a, b in zip(shells_mm[:-1], shells_mm[1:])]))


def line_points(lines: np.ndarray, spacing, depth_axis: int) -> np.ndarray:
    """Where each line row meets the lesion, in mm on the volume's own axes."""
    points = np.zeros((len(lines), 3))
    points[:, depth_axis] = lines[:, 0] * spacing[depth_axis]
    others = [a for a in range(3) if a != depth_axis]
    points[:, others[0]], points[:, others[1]] = lines[:, 1] * spacing[others[0]], lines[:, 2] * spacing[others[1]]
    return points


def line_surface_field(points: np.ndarray, lines: np.ndarray, mesh: trimesh.Trimesh, spectrum: dict, tau: float = FIELD_TAU, coverage: float = 0.1) -> np.ndarray:
    """
    The line excess carried to the surface: lines go to the nearest vertex, then the heat kernel spreads the sums over the surface.

    `points` must be on the mesh's frame. Vertices with too little line weight nearby, such as the lesion's top and bottom
    caps, which coronal planes do not surround, are nan.
    """
    lam, phi, mass = spectrum["lam"], spectrum["phi"], spectrum["mass"]
    vertex = cKDTree(mesh.vertices).query(points)[1]
    heat = np.exp(-lam * tau * mesh.area)
    excess = phi @ (heat * (phi.T @ np.bincount(vertex, lines[:, 5] * lines[:, 4], len(mass))))
    weight = phi @ (heat * (phi.T @ np.bincount(vertex, lines[:, 5], len(mass))))
    field = np.full(len(mass), np.nan)
    keep = weight > coverage * (mass @ weight) / mass.sum()
    field[keep] = excess[keep] / weight[keep]
    return field


def line_surface_summary(field: np.ndarray, height: np.ndarray, spectrum: dict, mesh: trimesh.Trimesh, tau: float = FIELD_TAU) -> dict:
    """Strongest tenth of the surface, share of it with lines running outward, and whether those sit on the surface's bumps."""
    lam, phi, mass = spectrum["lam"], spectrum["phi"], spectrum["mass"]
    keep = np.isfinite(field)
    bumps = phi @ (np.exp(-lam * tau * mesh.area) * (phi.T @ (mass * height)))
    return dict(radial_p90=float(weighted_quantiles(field[keep], mass[keep], np.array([0.9]))[0]),
                radial_share=float(mass[keep][field[keep] > 0].sum() / mass[keep].sum()),
                coupling=float(spearmanr(bumps[keep], field[keep])[0]), covered=float(mass[keep].sum() / mass.sum()))
