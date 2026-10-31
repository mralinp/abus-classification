import pytest
from scipy.special import sph_harm_y
from abus_classification.features.shape_descriptor.shape_descriptors import sphere_level_spread
import itertools

import scipy.sparse as sp
from abus_classification.features.shape_descriptor.shape_descriptors import heat_trace_coefficients
from scipy.spatial.transform import Rotation
from abus_classification import lb_invariance
from scipy.sparse.linalg import eigsh
from abus_classification.features.shape_descriptor.shape_descriptors import lb_spectrum
import numpy as np
import trimesh

from abus_classification.features.shape_descriptor import (
    boundary_signature2d,
    boundary_signature3d,
    compute_eigendecomposition,
    compute_laplace_beltrami_operator,
)


def test_boundary_signature_2d():
    x = np.array([[1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 1],
                  [1, 0, 0, 0, 1],
                  [1, 0, 0, 0, 1],
                  [1, 1, 1, 1, 1]], dtype=np.uint8) * 255
    y = np.array([2., 2.828427, 2., 2.828427, 2., 2.828427, 2., 2.828427], dtype=np.float32)

    sig = boundary_signature2d(x, resolution=45)

    np.testing.assert_allclose(sig, y, rtol=1e-5)


def test_boundary_signature_2d_distinguishes_vertical_from_horizontal():
    # A tall rectangle outline: 5 from the centre along the rows, 2 along the columns.
    x = np.zeros((11, 5), dtype=np.uint8)
    x[[0, -1], :] = 1
    x[:, [0, -1]] = 1

    sig = boundary_signature2d(x, resolution=90)

    np.testing.assert_allclose(sig, [5., 2., 5., 2.], rtol=1e-5)


def test_boundary_signature_3d():
    x = np.zeros((5, 5, 5), dtype=np.uint8)
    x[1:4, 1:4, 1:4] = 1

    res = boundary_signature3d(x, resolution=(15, 15))

    assert res.shape == (24, 24)
    filled = res[res > 0]
    assert filled.size > 0
    assert filled.min() >= 1.0 - 1e-6
    assert filled.max() <= np.sqrt(3) + 1e-6


def test_laplace_beltrami_spectrum_of_the_unit_sphere():
    # The Laplace-Beltrami eigenvalues of the unit sphere are l(l + 1):
    # 0 once, 2 three times, 6 five times.
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    L, M = compute_laplace_beltrami_operator(mesh)
    eigenvalues, _ = compute_eigendecomposition(L, M, k=9)

    assert abs(eigenvalues[0]) < 1e-3
    np.testing.assert_allclose(eigenvalues[1:4], 2.0, rtol=0.05)
    np.testing.assert_allclose(eigenvalues[4:9], 6.0, rtol=0.05)


def dimpled_sphere(cut=0.6):
    """Sphere and the same sphere with the cap above z = cut mirrored inward: an isometry, not a rigid motion."""
    n_rings, n_around = 24, 48
    angles = np.arccos(cut) * np.arange(n_rings // 2 + 1) / (n_rings // 2)   # the cut is a ring of vertices
    angles = np.concatenate([angles, np.linspace(angles[-1], np.pi, n_rings // 2 + 1)[1:]])
    theta = 2 * np.pi * np.arange(n_around) / n_around
    rings = np.array([[(np.sin(a) * np.cos(t), np.sin(a) * np.sin(t), np.cos(a)) for t in theta]
                      for a in angles[1:-1]]).reshape(-1, 3)
    vertices = np.vstack([[0, 0, 1], rings, [0, 0, -1]])
    ring = lambda r, i: 1 + r * n_around + i % n_around
    faces = [(0, ring(0, i), ring(0, i + 1)) for i in range(n_around)]
    for r in range(len(angles) - 3):
        for i in range(n_around):
            faces += [(ring(r, i), ring(r + 1, i), ring(r + 1, i + 1)), (ring(r, i), ring(r + 1, i + 1), ring(r, i + 1))]
    last = len(vertices) - 1
    faces += [(ring(len(angles) - 3, i), last, ring(len(angles) - 3, i + 1)) for i in range(n_around)]
    sphere = trimesh.Trimesh(vertices, faces, process=False)
    dimpled = vertices.copy()
    above = dimpled[:, 2] > cut
    dimpled[above, 2] = 2 * cut - dimpled[above, 2]
    return sphere, trimesh.Trimesh(dimpled, faces, process=False)


def test_spectrum_ignores_rigid_motion():
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh.vertices = mesh.vertices * [1.0, 0.7, 0.4]
    moved = mesh.copy()
    moved.apply_transform(trimesh.transformations.random_rotation_matrix(np.random.default_rng(0).random(3)))
    moved.apply_translation([3.0, -2.0, 5.0])
    np.testing.assert_allclose(lb_spectrum(moved, 10)[1:], lb_spectrum(mesh, 10)[1:], rtol=1e-8)


def test_area_normalised_spectrum_ignores_scale():
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh.vertices = mesh.vertices * [1.0, 0.7, 0.4]
    scaled = mesh.copy()
    scaled.apply_scale(7.0)
    np.testing.assert_allclose(lb_spectrum(scaled, 10)[1:], lb_spectrum(mesh, 10)[1:], rtol=1e-6)
    raw, raw_scaled = lb_spectrum(mesh, 10, False)[1:], lb_spectrum(scaled, 10, False)[1:]
    np.testing.assert_allclose(raw_scaled, raw / 49, rtol=1e-6)


def test_spectrum_cannot_see_an_isometric_dent_but_sees_a_stretch():
    sphere, dimpled = dimpled_sphere()
    assert dimpled.volume < 0.95 * sphere.volume           # a different solid...
    np.testing.assert_allclose(dimpled.area, sphere.area)  # ...with the same surface...
    np.testing.assert_allclose(lb_spectrum(dimpled, 12)[1:], lb_spectrum(sphere, 12)[1:], rtol=1e-6)  # ...and spectrum

    stretched = sphere.copy()
    stretched.vertices = stretched.vertices * [1.0, 1.0, 1.5]
    assert np.abs(lb_spectrum(stretched, 12)[1:] / lb_spectrum(sphere, 12)[1:] - 1).max() > 0.05


def test_eigenvalues_of_l_alone_follow_the_vertex_count():
    # With a nearly uniform mass matrix M = (area / n) I, so eig(L) = lambda * area / n: finer meshes shrink them.
    coarse, fine = (trimesh.creation.icosphere(subdivisions=s) for s in (3, 4))
    eig_l = lambda m: np.sort(eigsh(compute_laplace_beltrami_operator(m)[0], k=4, sigma=-1e-8, which="LM",
                                    return_eigenvectors=False))[1:]
    np.testing.assert_allclose(eig_l(coarse) / eig_l(fine), len(fine.vertices) / len(coarse.vertices), rtol=0.05)


def test_voxel_pipeline_keeps_the_spectrum_of_a_moved_ball():
    grid = np.ogrid[-16:17, -16:17, -16:17]
    ball = sum(g.astype(float) ** 2 for g in grid) <= 14 ** 2
    moved = lb_invariance.move(ball.astype(np.float32), Rotation.from_euler("xyz", [20, 50, 70], degrees=True),
                               np.array([0.3, -0.2, 0.4]))
    lam, _, area, *_ = lb_invariance.measure(moved > 0.5, 1.0, "smoothed")
    # A sphere has lambda * area = l(l + 1) * 4 pi: 8 pi three times.
    np.testing.assert_allclose(lam[1:4] * area, 8 * np.pi, rtol=0.03)


def test_heat_trace_recovers_area_and_euler_number_of_the_sphere():
    lam = np.concatenate([[l * (l + 1)] * (2 * l + 1) for l in range(40)])   # analytic spectrum
    c0, chi, c2 = heat_trace_coefficients(lam, 4 * np.pi)
    assert abs(c0 - 1) < 1e-3 and abs(chi - 2) < 0.01
    assert abs(c2 - (4 * np.pi) ** 2 / (60 * np.pi)) < 0.05  # int K^2 dA = 4 pi


def test_heat_trace_tells_a_torus_from_a_sphere_on_fine_meshes():
    sphere = trimesh.creation.icosphere(subdivisions=6)
    torus = trimesh.creation.torus(major_radius=2, minor_radius=1, major_sections=256, minor_sections=128)
    assert abs(heat_trace_coefficients(lb_spectrum(sphere, 100, False), sphere.area)[1] - 2) < 0.1
    assert abs(heat_trace_coefficients(lb_spectrum(torus, 100, False), torus.area)[1]) < 0.15


# Mirror-gluing patterns of 7 tiles: three involutions of GL(3, 2), acting on the points and on the lines of the Fano
# plane. These actions are almost conjugate, so by Sunada's theorem the glued surfaces share a spectrum.
SUNADA_POINTS = ([3, 1, 5, 0, 4, 2, 6], [0, 3, 4, 1, 2, 5, 6], [0, 2, 1, 3, 4, 6, 5])
SUNADA_LINES = ([3, 1, 5, 0, 4, 2, 6], [0, 3, 4, 1, 2, 5, 6], [2, 1, 0, 3, 6, 5, 4])


def sunada_surface(gluing, n=12):
    """L and M of a closed flat surface: 14 copies of one triangle, two sheets joined wherever a side has no partner."""
    corners = np.array([[0, 0, 0], [1, 0, 0], [0.3, 0.8, 0]], dtype=float)
    bary = [(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)]
    index = {b: k for k, b in enumerate(bary)}
    faces = []
    for i in range(n):
        for j in range(n - i):
            faces.append((index[i, j, n - i - j], index[i + 1, j, n - i - j - 1], index[i, j + 1, n - i - j - 1]))
            if i + j < n - 1:
                faces.append((index[i + 1, j, n - i - j - 1], index[i + 1, j + 1, n - i - j - 2],
                              index[i, j + 1, n - i - j - 1]))
    tile = trimesh.Trimesh((np.array(bary) / n) @ corners, faces, process=False)
    L, M = compute_laplace_beltrami_operator(tile)
    size = len(bary)
    parent = np.arange(14 * size)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for side, partner in enumerate(gluing):
        on_side = [k for k, b in enumerate(bary) if b[side] == 0]
        for tile_a, tile_b in enumerate(partner):
            for k in on_side:
                if tile_a == tile_b:
                    parent[find(2 * tile_a * size + k)] = find((2 * tile_a + 1) * size + k)
                else:
                    for sheet in (0, 1):
                        parent[find((2 * tile_a + sheet) * size + k)] = find((2 * tile_b + sheet) * size + k)
    _, label = np.unique([find(x) for x in range(14 * size)], return_inverse=True)
    coo, count = L.tocoo(), label.max() + 1
    rows = np.concatenate([label[c * size + coo.row] for c in range(14)])
    cols = np.concatenate([label[c * size + coo.col] for c in range(14)])
    glued_l = sp.csr_matrix((np.tile(coo.data, 14), (rows, cols)), shape=(count, count))
    mass = np.bincount(label, np.tile(M.diagonal(), 14), minlength=count)
    return glued_l, sp.diags(mass).tocsr()


def test_sunada_pair_shares_a_spectrum_but_is_not_isometric():
    (L1, M1), (L2, M2) = sunada_surface(SUNADA_POINTS), sunada_surface(SUNADA_LINES)
    lam1, phi1 = compute_eigendecomposition(L1, M1, 40)
    lam2, phi2 = compute_eigendecomposition(L2, M2, 40)
    np.testing.assert_allclose(lam1[1:], lam2[1:], rtol=1e-10)

    relabelled = any(all([s[p[i]] for i in range(7)] == [q[s[i]] for i in range(7)]
                         for p, q in zip(SUNADA_POINTS, SUNADA_LINES)) for s in itertools.permutations(range(7)))
    assert not relabelled
    # The heat kernel diagonal over the surface is an isometry invariant, and its values differ.
    t = 1 / lam1[3]
    h1, h2 = (np.sort((np.exp(-lam * t) * phi ** 2).sum(1)) for lam, phi in ((lam1, phi1), (lam2, phi2)))
    assert np.abs(h1 - h2).max() / h1.mean() > 0.01


def lobed_sphere(degree, epsilon, subdivisions=6):
    """Unit sphere with radius 1 + epsilon * Y, where Y is the axial spherical harmonic scaled to a peak of 1."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    theta = np.arccos(np.clip(mesh.vertices[:, 2], -1, 1))
    y = sph_harm_y(degree, 0, theta, 0 * theta).real
    return trimesh.Trimesh(mesh.vertices * (1 + epsilon * y / np.abs(y).max())[:, None], mesh.faces, process=False)


def predicted_level_spread(level, degree, epsilon):
    """First-order std / mean of level l under that deformation: delta lambda_m = -2 lambda int f |Y_lm|^2 dA."""
    x, w = np.polynomial.legendre.leggauss(200)
    theta = np.arccos(x)
    peak = np.abs(sph_harm_y(degree, 0, np.linspace(0, np.pi, 4001), 0.0).real).max()
    f = epsilon * sph_harm_y(degree, 0, theta, 0.0).real / peak
    shifts = [-2 * 2 * np.pi * (w * f * np.abs(sph_harm_y(level, m, theta, 0.0)) ** 2).sum() for m in range(-level, level + 1)]
    return np.std(shifts)   # shifts are relative to lambda


def test_hersch_bound_is_reached_by_the_sphere_only():
    sphere = trimesh.creation.icosphere(subdivisions=5)
    assert abs(lb_spectrum(sphere, 4)[1] / (8 * np.pi) - 1) < 1e-3
    for scale in ([1, 1, 2], [1, 1, 0.5], [1, 0.7, 0.4]):
        stretched = sphere.copy()
        stretched.vertices = stretched.vertices * scale
        assert lb_spectrum(stretched, 4)[1] < 0.95 * 8 * np.pi


@pytest.mark.parametrize("degree", [2, 4, 6, 8])
def test_even_lobes_split_exactly_the_levels_the_selection_rule_allows(degree):
    spread = sphere_level_spread(lb_spectrum(lobed_sphere(degree, 0.02), 49, False))
    first = degree // 2
    assert all(spread[:first - 1] < 0.25 * spread[first - 1])   # lower levels split only at second order
    assert spread[first - 1] == pytest.approx(predicted_level_spread(first, degree, 0.02), rel=0.1)


def test_odd_lobes_split_only_at_second_order():
    small, large = (sphere_level_spread(lb_spectrum(lobed_sphere(3, e), 16, False), levels=3)[0] for e in (0.02, 0.05))
    assert 4 < large / small < 9     # (2.5)^2 = 6.25, not 2.5
