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
