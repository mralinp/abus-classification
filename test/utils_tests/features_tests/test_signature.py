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
