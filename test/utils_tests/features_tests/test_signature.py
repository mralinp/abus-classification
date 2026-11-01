from sklearn.metrics import roc_auc_score
from abus_classification import lb_evaluation
from abus_classification import lb_classification
from abus_classification.features.shape_descriptor.shape_descriptors import TAUS, spectral_summaries
from abus_classification.features.shape_descriptor.spharm import map_quality, real_harmonics, spharm_coefficients, spharm_power, sphere_map
from scipy import ndimage
from abus_classification.features.shape_descriptor.shape_descriptors import spectrum_bound
from abus_classification.utils.mesh import get_mesh_from_3d_mask
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


def test_spectrum_stays_inside_the_metric_bound():
    rng = np.random.default_rng(0)
    sphere = trimesh.creation.icosphere(subdivisions=4)
    lam = lb_spectrum(sphere, 30, False)[1:]
    deformations = [sphere.vertices * [1, 1, 1.3], sphere.vertices * 1.7,
                    sphere.vertices * (1 + 0.1 * np.sin(5 * sphere.vertices[:, [2]])),
                    sphere.vertices + rng.normal(scale=0.002, size=sphere.vertices.shape)]
    for vertices in deformations:
        moved = trimesh.Trimesh(vertices, sphere.faces, process=False)
        lower, upper = spectrum_bound(sphere, moved)
        ratio = lb_spectrum(moved, 30, False)[1:] / lam
        assert lower * (1 - 1e-9) <= ratio.min() and ratio.max() <= upper * (1 + 1e-9)


def test_bound_ignores_rigid_motion_and_is_tight_for_scale():
    sphere = trimesh.creation.icosphere(subdivisions=3)
    moved = sphere.copy()
    moved.apply_transform(trimesh.transformations.rotation_matrix(1.0, [1, 2, 3]))
    np.testing.assert_allclose(spectrum_bound(sphere, moved), (1, 1), atol=1e-8)
    scaled = trimesh.Trimesh(sphere.vertices * 2, sphere.faces, process=False)
    np.testing.assert_allclose(spectrum_bound(sphere, scaled), (0.25, 0.25), atol=1e-8)


def test_low_modes_ignore_fine_bumps_that_widen_the_bound():
    round_sphere = trimesh.creation.icosphere(subdivisions=6)
    reference = lb_spectrum(round_sphere, 21)[1:]
    change = lambda mesh: np.abs(lb_spectrum(mesh, 21)[1:] / reference - 1).mean()
    coarse, fine = lobed_sphere(2, 0.02), lobed_sphere(16, 0.02)
    assert change(fine) < 0.25 * change(coarse)                  # same amplitude
    assert np.ptp(spectrum_bound(round_sphere, fine)) > np.ptp(spectrum_bound(round_sphere, coarse))


def dumbbell(neck_mm, h=0.25):
    """Two unit-radius-scale balls joined by a cylinder: a small change at the neck, a large one in lambda_1."""
    x, y, z = np.mgrid[-6:6:h, -3:3:h, -3:3:h]
    left, right = (x + 3) ** 2 + y ** 2 + z ** 2 <= 4, (x - 3) ** 2 + y ** 2 + z ** 2 <= 4
    neck = (np.abs(x) <= 3) & (y ** 2 + z ** 2 <= neck_mm ** 2)
    field = ndimage.gaussian_filter((left | right | neck).astype(np.float32), 1)
    return get_mesh_from_3d_mask(np.pad(field, 2), spacing=(h, h, h), preprocess=False)


def test_a_thin_neck_collapses_lambda_1_but_barely_changes_the_volume():
    thick, thin = dumbbell(1.0), dumbbell(0.4)
    assert thin.is_watertight and thin.body_count == 1
    assert abs(thin.volume / thick.volume - 1) < 0.1
    assert lb_spectrum(thin, 3)[1] < 0.5 * lb_spectrum(thick, 3)[1]


def test_sphere_map_is_identity_like_on_a_sphere_and_reconstructs_it():
    mesh = trimesh.creation.icosphere(subdivisions=4)
    sphere = sphere_map(mesh)
    flipped, stretch, _ = map_quality(mesh, sphere)
    assert flipped == 0 and stretch < 1.01
    coefficients = spharm_coefficients(mesh, sphere, 6)
    assert np.linalg.norm(real_harmonics(sphere, 6) @ coefficients - mesh.vertices, axis=1).max() < 0.01


def test_real_harmonics_are_orthonormal_with_the_laplace_beltrami_eigenvalues():
    mesh = trimesh.creation.icosphere(subdivisions=6)
    L, M = compute_laplace_beltrami_operator(mesh)
    basis = real_harmonics(mesh.vertices, 4)
    gram = basis.T @ (M @ basis)
    np.testing.assert_allclose(gram, np.eye(25), atol=1e-2)
    expected = np.concatenate([[l * (l + 1)] * (2 * l + 1) for l in range(5)])
    np.testing.assert_allclose(np.diag(basis.T @ (L @ basis)), expected, atol=0.1)


def test_power_ignores_rotating_the_shape_and_scales_with_area():
    mesh = lobed_sphere(4, 0.2, subdivisions=4)
    power = spharm_power(spharm_coefficients(mesh, sphere_map(mesh), 8), 8)
    moved = mesh.copy()
    moved.apply_transform(trimesh.transformations.rotation_matrix(1.0, [1, 2, 3]))
    turned = spharm_power(spharm_coefficients(moved, sphere_map(moved), 8), 8)
    np.testing.assert_allclose(turned[1:], power[1:], rtol=0.01, atol=1e-5)


@pytest.mark.parametrize("degree", [2, 4, 6])
def test_a_lobe_of_degree_d_shows_at_degrees_d_plus_and_minus_one(degree):
    mesh = lobed_sphere(degree, 0.05, subdivisions=5)
    power = spharm_power(spharm_coefficients(mesh, sphere_map(mesh), 8), 8)
    others = [l for l in range(2, 9) if l not in (degree - 1, degree + 1)]
    assert power[degree + 1] > 10 * power[others].max()


def test_degree_one_axes_order_ellipsoids_but_compress_their_proportions():
    # An ellipsoid is not linear in a conformal parametrisation, unlike the area-preserving one of classic SPHARM.
    def ratio(stretch):
        mesh = trimesh.creation.icosphere(subdivisions=5)
        mesh.vertices = mesh.vertices * [1, 1, stretch]
        axes = np.linalg.svd(spharm_coefficients(mesh, sphere_map(mesh), 6)[1:4], compute_uv=False)
        return axes[0] / axes[2]
    mild, strong = ratio(1.5), ratio(2.5)
    assert 1 < mild < 1.5 and mild < strong < 2.5


def bumpy_sphere(bumps=14, seed=3):
    mesh = trimesh.creation.icosphere(subdivisions=5)
    centres = np.random.default_rng(seed).normal(size=(bumps, 3))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    radius = 1 + 0.45 * np.exp((mesh.vertices @ centres.T - 1) / 0.012).sum(1)
    return trimesh.Trimesh(mesh.vertices * radius[:, None], mesh.faces, process=False)


def test_summaries_ignore_rotation_and_scale():
    mesh = bumpy_sphere()
    base = spectral_summaries(mesh, 60)
    moved = mesh.copy()
    moved.apply_transform(trimesh.transformations.rotation_matrix(1.0, [1, 2, 3]))
    moved.apply_scale(3.0)
    other = spectral_summaries(moved, 60)
    for key in ("hks", "wks", "gps"):
        np.testing.assert_allclose(other[key], base[key], rtol=1e-3, atol=1e-6)
    np.testing.assert_array_equal(other["tips"], base["tips"])


def test_mean_heat_signature_is_the_heat_trace():
    mesh = trimesh.creation.icosphere(subdivisions=4)
    lam = lb_spectrum(mesh, 200)
    summary = spectral_summaries(mesh, 200)
    np.testing.assert_allclose(summary["hks"][0], np.exp(-np.outer(TAUS, lam)).sum(1), rtol=1e-6)


def test_heat_maxima_count_protrusions_and_a_sphere_has_no_spread():
    bumps = spectral_summaries(bumpy_sphere(), 200)
    assert 10 <= bumps["tips"][-1] <= 16        # 14 bumps, at the longest time
    sphere = spectral_summaries(trimesh.creation.icosphere(subdivisions=5), 200)
    assert (sphere["hks"][1] / sphere["hks"][0] < 0.02).all() and (bumps["hks"][1, :5] / bumps["hks"][0, :5] > 0.05).all()


def test_mean_wave_signature_is_always_one():
    # Its weights sum to 1 and phi^2 * area averages 1 over the surface, so only its spread can carry information.
    np.testing.assert_allclose(spectral_summaries(bumpy_sphere(), 200)["wks"][0], 1, atol=1e-6)


def test_signature_spread_tells_isospectral_surfaces_apart_where_the_spectrum_cannot():
    (L1, M1), (L2, M2) = sunada_surface(SUNADA_POINTS), sunada_surface(SUNADA_LINES)
    spreads, lam1, lam2 = [], compute_eigendecomposition(L1, M1, 40)[0], compute_eigendecomposition(L2, M2, 40)[0]
    for L, M in ((L1, M1), (L2, M2)):
        lam, phi = compute_eigendecomposition(L, M, 40)
        heat = (np.exp(-lam / lam[3]) * phi ** 2).sum(1)
        spreads.append(np.sqrt(M.diagonal() @ (heat - M.diagonal() @ heat / M.diagonal().sum()) ** 2))
    np.testing.assert_allclose(lam1[1:], lam2[1:], rtol=1e-10)
    assert abs(spreads[0] / spreads[1] - 1) > 1e-3


def test_classification_table_excludes_test_and_selection_ignores_noise():
    table = lb_classification.tv_table()
    assert len(table) == 130 and (table.split != "Test").all()
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(60, int), np.ones(70, int)]
    signal = (y + rng.normal(scale=0.6, size=130))[:, None]
    X = np.c_[signal, rng.normal(size=(130, 5))]
    chosen = lb_classification.forward({"signal": [0], **{f"noise {i}": [i + 1] for i in range(5)}}, X, y, 0)
    assert chosen[0] == "signal"


def test_evaluation_probability_matches_each_classifier_decision():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(40, int), np.ones(40, int)]
    X = np.c_[y + rng.normal(scale=0.8, size=80), rng.normal(size=(80, 3))]
    for name in ("Logistic C=1", "SVM linear C=0.1", "SVM RBF C=1", "Random forest"):
        model = lb_evaluation.make_model(dict(classifier=name, k=2)).fit(X, y)
        p = lb_evaluation.probability(model, X)
        assert ((p >= 0.5) == (model.predict(X) == 1)).mean() > 0.97 and roc_auc_score(y, p) > 0.7


from abus_classification.features.shape_descriptor.malignancy_signs import (
    area_budget, beam_orientation, line_field, line_points, line_summary, line_surface_field, mean_curvature, mesh_spectrum, outward_normals, radial_lines,
    spectral_moments, spectral_roundness, surface_height, surface_peaks, surface_signs)


def _bumps(directions, height=0.2, width=0.2, subdivisions=4):
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
    unit = sphere.vertices / np.linalg.norm(sphere.vertices, axis=1, keepdims=True)
    gain = sum(height * np.exp(-np.linalg.norm(unit - np.asarray(d) / np.linalg.norm(d), axis=1) ** 2 / (2 * width ** 2)) for d in directions)
    return trimesh.Trimesh(unit * (1 + gain)[:, None], sphere.faces, process=False)


def test_the_cotangent_laplacian_spends_exactly_twice_the_area_on_any_surface():
    mesh = _bumps([(1, 0, 0), (0, 1, 1)])
    L, _ = compute_laplace_beltrami_operator(mesh)
    assert np.isclose(np.trace(mesh.vertices.T @ (L @ mesh.vertices)), 2 * mesh.area)
    s = mesh_spectrum(mesh, 400)
    assert 0.98 < spectral_moments(s["lam"], s["c"])[1] / (2 * mesh.area) <= 1.0     # the modes kept hold nearly all of it


def test_bending_of_a_sphere_and_a_spheroid_matches_the_integral_of_squared_mean_curvature():
    sphere = trimesh.creation.icosphere(subdivisions=4)
    s = mesh_spectrum(sphere, 100)
    assert np.isclose(spectral_moments(s["lam"], s["c"])[2] / 4, 4 * np.pi, rtol=0.02)
    assert np.allclose(mean_curvature(s["lam"], s["phi"], s["c"]), 1.0, rtol=0.03)
    a, b = 1.0, 2.0
    u = np.linspace(0, np.pi, 20001)
    w = np.sqrt(a ** 2 * np.cos(u) ** 2 + b ** 2 * np.sin(u) ** 2)
    exact = 2 * np.pi * np.trapezoid(((a * b / w ** 3 + b / (a * w)) / 2) ** 2 * a * np.sin(u) * w, u)
    spheroid = trimesh.Trimesh(sphere.vertices * [a, a, b], sphere.faces, process=False)
    s = mesh_spectrum(spheroid, 300)
    assert np.isclose(spectral_moments(s["lam"], s["c"])[2] / 4, exact, rtol=0.05)


def test_spectral_roundness_is_one_only_for_a_sphere_and_ignores_pose_and_scale():
    sphere = trimesh.creation.icosphere(subdivisions=4)
    def roundness(mesh):
        s = mesh_spectrum(mesh, 150)
        return spectral_roundness(s["lam"], s["c"])
    assert roundness(sphere) > 0.999
    assert roundness(trimesh.Trimesh(sphere.vertices * [1, 1, 1.5], sphere.faces, process=False)) < 0.95
    bumpy = _bumps([(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, 0)])
    moved = trimesh.Trimesh(3.0 * bumpy.vertices @ trimesh.transformations.random_rotation_matrix(np.random.default_rng(0).random(3))[:3, :3] + 5.0, bumpy.faces, process=False)
    assert roundness(bumpy) < 0.9
    assert np.isclose(roundness(moved), roundness(bumpy), rtol=1e-6)


def test_area_budget_moves_from_low_to_high_degrees_as_the_surface_gets_bumpier():
    sphere = trimesh.creation.icosphere(subdivisions=4)
    s = mesh_spectrum(sphere, 200)
    budget = area_budget(s["lam"], s["c"], sphere.area)
    assert budget[0] > 0.999 and budget[1:].sum() < 1e-3
    many = _bumps(trimesh.creation.icosphere(subdivisions=1).vertices, height=0.08, width=0.12)
    s = mesh_spectrum(many, 200)
    assert area_budget(s["lam"], s["c"], many.area)[3:].sum() > 0.02


def test_peaks_count_the_lobes_and_ignore_small_ripples():
    directions = [(1, 0, 0), (0, 1, 0), (-1, -1, 1)]
    mesh = _bumps(directions, height=0.15, width=0.25)
    assert (outward_normals(mesh) * mesh.vertices).sum(1).min() > 0
    s = mesh_spectrum(mesh, 200)
    height = surface_height(mesh, s["phi"], s["c"], body_modes=4)
    vertices, prominence = surface_peaks(height, mesh)
    assert (prominence >= 0.05).sum() == 3
    tops = mesh.vertices[vertices[prominence >= 0.05]]
    tops /= np.linalg.norm(tops, axis=1, keepdims=True)
    wanted = np.array(directions) / np.linalg.norm(directions, axis=1, keepdims=True)
    assert (tops @ wanted.T).max(axis=0).min() > 0.97
    ripple = 0.01 * np.random.default_rng(0).normal(size=len(height))
    assert (surface_peaks(height + ripple, mesh)[1] >= 0.05).sum() == 3
    assert len(surface_peaks(height + ripple, mesh)[0]) > 3 * 5                  # a plain count of maxima is fooled


def test_peak_prominence_on_a_path_of_values_by_hand():
    # A triangle strip, with vertex values along it 0, 3, 1, 5, 2: peaks at 5 (whole range) and 3 (3 falls to the saddle at 1).
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [2, 9, 0], [2, -9, 0]], float)
    faces = np.array([[i, i + 1, 5] for i in range(4)] + [[i + 1, i, 6] for i in range(4)])
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    values = np.array([0, 3, 1, 5, 2, -9, -9], float)
    peaks, prominence = surface_peaks(values, mesh)
    assert dict(zip(peaks.tolist(), prominence.tolist())) == {3: 14.0, 1: 2.0}


def test_beam_orientation_reads_a_taller_than_wide_shape_and_needs_the_axis():
    sphere = trimesh.creation.icosphere(subdivisions=3)
    def orient(scale):
        s = mesh_spectrum(trimesh.Trimesh(sphere.vertices * scale, sphere.faces, process=False), 40)
        return beam_orientation(s["c"], 1)
    assert np.isclose(orient([1, 1, 1])[0], 1 / 3, rtol=0.01)
    tall, wide = orient([1, 2, 1]), orient([2, 1, 1])
    assert tall[0] > 0.5 and tall[1] > 85
    assert wide[0] < 0.2 and wide[1] < 5
    assert not np.isclose(tall[0], wide[0])                                       # the spectrum alone cannot tell these two apart


def _lines_image(kind, shape=(101, 11, 101), pixel=0.6, radius_mm=8.0):
    # Bright lines around a disc in coronal planes (rows, cols); axis 1 is the beam.
    rows, cols = np.mgrid[:shape[0], :shape[2]].astype(float)
    r, a = np.hypot(rows - 50, cols - 50) * pixel, np.arctan2(cols - 50, rows - 50)
    if kind == "radial":
        plane = np.cos(18 * a) > 0.8
    elif kind == "wrapping":
        plane = np.cos(2 * np.pi * r / 4.0) > 0.8
    elif kind == "noise":
        plane = None
    volume = np.zeros(shape, np.float32)
    rng = np.random.default_rng(0)
    for k in range(shape[1]):
        volume[:, k] = ndimage.gaussian_filter(rng.normal(size=(shape[0], shape[2])), 1.2) if plane is None else ndimage.gaussian_filter((plane & (r > radius_mm)).astype(np.float32), 0.8)
    mask = np.zeros(shape, bool)
    mask[:, 3:8] = (r < radius_mm)[:, None]
    return volume, mask, (pixel,) * 3


@pytest.mark.parametrize("kind, low, high", [("radial", 0.1, 1), ("wrapping", -1, -0.1), ("noise", -0.03, 0.03)])
def test_radial_lines_tell_lines_running_out_from_lines_wrapping_around(kind, low, high):
    volume, mask, spacing = _lines_image(kind)
    theta, strength = line_field(volume, spacing, 1)
    lines = radial_lines(theta, strength, mask, spacing, 1)
    assert low < line_summary(lines, 5)["radial"] < high


def test_radial_lines_cancel_a_shared_orientation_and_reach_the_surface():
    volume, mask, spacing = _lines_image("noise")
    stripes = np.cos(2 * np.pi * np.arange(volume.shape[2]) * spacing[2] / 3.0)[None, None] * np.ones(volume.shape, np.float32)
    theta, strength = line_field(stripes, spacing, 1)
    assert abs(line_summary(radial_lines(theta, strength, mask, spacing, 1), 5)["radial"]) < 0.01
    volume, mask, spacing = _lines_image("radial")
    theta, strength = line_field(volume, spacing, 1)
    lines = radial_lines(theta, strength, mask, spacing, 1)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=8.0)
    sphere.vertices += np.array([50, 5, 50]) * 0.6
    s = mesh_spectrum(sphere, 150)
    field = line_surface_field(line_points(lines, spacing, 1), lines, sphere, s)
    equator = np.abs(sphere.vertices[:, 1] - 3.0) < 2.0
    assert np.nanmin(field[equator]) > 0.1                                         # lines at the equator, none at the caps
    assert np.isnan(field[np.abs(sphere.vertices[:, 1] - 3.0) > 7.5]).all()


def test_surface_signs_ignore_scale_and_stay_finite():
    mesh = _bumps([(1, 0, 0), (0, 1, 0)])
    a = surface_signs(mesh, mesh_spectrum(mesh, 120), 1)
    big = trimesh.Trimesh(4.0 * mesh.vertices, mesh.faces, process=False)
    b = surface_signs(big, mesh_spectrum(big, 120), 1)
    for key in a:
        assert np.allclose(a[key], b[key], rtol=1e-6)


def test_sham_lesions_stay_in_the_crop_and_clear_the_lesion_and_its_shell():
    from abus_classification.lb_malignancy import sham_masks
    mask = np.zeros((300, 30, 300), bool)
    mask[140:160, 10:20, 140:160] = True
    shams = sham_masks(mask, (0.6, 0.6, 0.6))
    assert len(shams) == 4
    for sham in shams:
        assert sham.sum() == mask.sum()
        distance = ndimage.distance_transform_edt(~sham, sampling=0.6)
        assert distance[mask].min() >= 20.0                                        # 10 mm shells around each do not touch
    assert len(sham_masks(np.roll(mask, 70, axis=0), (0.6, 0.6, 0.6))) == 3        # no room on one side


def test_one_feature_per_family_is_chosen_only_from_the_lesions_given():
    import pandas as pd
    from abus_classification.lb_malignancy import family_picks
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    part = np.arange(100)
    good_here = np.where(np.arange(200) < 100, y + 0.2 * rng.normal(size=200), rng.normal(size=200))
    good_elsewhere = np.where(np.arange(200) >= 100, y + 0.2 * rng.normal(size=200), rng.normal(size=200))
    rows = pd.DataFrame({"y": y, "Size: log volume": rng.normal(size=200), "F: here": good_here, "F: elsewhere": good_elsewhere})
    assert family_picks(rows, {"F": ["F: here", "F: elsewhere"]}, ["F"], part) == ["F: here"]
    assert family_picks(rows, {"F": ["F: here", "F: elsewhere"]}, ["F"], part + 100) == ["F: elsewhere"]


def test_stored_model_scores_exactly_as_the_fitted_one():
    import pandas as pd
    from abus_classification.lb_classification import linear
    from abus_classification.lb_malignancy import decision, fit_model
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(5, 3, 80)})
    rows["y"] = (rows.a + rows.b / 3 + rng.normal(size=80) > 1.5).astype(int)
    fresh = pd.DataFrame({"a": rng.normal(size=20), "b": rng.normal(5, 3, 20)})
    fitted = linear().fit(rows[["a", "b"]].values, rows.y.values)
    assert np.allclose(decision(fit_model(rows, ["a", "b"]), fresh), fitted.decision_function(fresh[["a", "b"]].values))
