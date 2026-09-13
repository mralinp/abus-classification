"""
Coronal spiculation after Tan et al.

In automated 3D breast ultrasound, malignant lesions often show spiculation in
coronal planes (parallel to the skin) that is invisible in handheld 2D
ultrasound. Tan et al. measure it with the stellate-lesion statistic that
Karssemeijer and te Brake developed for mammography, computed in 2D in every
coronal plane:

1. A line orientation theta_j and strength W_j are estimated at every voxel by
   maximising the directional second-order Gaussian derivative.
2. For a site i, voxels j at in-plane distance r_ij in [r_min, r_max] with W_j
   above a small threshold form the neighbourhood N_i.
3. Voxel j points at i if |angle(i - j) - theta_j| < R / r_ij (eq. 2.4). The
   chance of that happening for a random orientation is p_j = 2R / (pi r_ij)
   (eq. 2.3).
4. With x_j = 1 - p_j when j points at i and -p_j otherwise (eq. 2.2), the
   normalised measure is the binomial z-score
       f = sum(x_j) / sqrt(sum(p_j (1 - p_j)))                     (eq. 2.5)
5. f is computed for 12 mm < r_max < 30 mm with r_min = R = 3 mm, and the
   maximum f_m is the spiculation value of the voxel.

Equation numbers refer to chapter 2 of Tan's thesis, which reproduces the
2012 IEEE TMI paper.

References:
    T. Tan, B. Platel, H. Huisman, C. I. Sanchez, R. Mus, N. Karssemeijer.
    Computer-aided lesion diagnosis in automated 3-D breast ultrasound using
    coronal spiculation. IEEE Transactions on Medical Imaging 31(5):1034-1042,
    2012.

    T. Tan. Automated 3D breast ultrasound image analysis. PhD thesis, Radboud
    University Nijmegen, 2014 (chapters 2, 3 and 5).

    N. Karssemeijer, G. M. te Brake. Detection of stellate distortions in
    mammograms. IEEE Transactions on Medical Imaging 15(5):611-619, 1996.
"""
from typing import Sequence, Tuple

import numpy as np
from scipy import fft, ndimage


def line_orientation(planes: np.ndarray,
                     sigmas_px: Sequence[float] = (1.5,),
                     polarity: str = "bright") -> Tuple[np.ndarray, np.ndarray]:
    """
    Line orientation and strength in every 2D plane of a stack.

    The directional second-order Gaussian derivative is maximised over
    direction in closed form from the Hessian. For bright lines (the "white
    lines" Tan describes in coronal planes) the strongest response is the most
    negative curvature, measured across the line; the line runs perpendicular
    to it. With several scales, each pixel keeps the scale of strongest
    scale-normalised response.

    Args:
        planes: Array of shape (n, rows, cols); each [k] is one 2D plane.
        sigmas_px: Gaussian scales in pixels.
        polarity: "bright" for light lines on a darker background, "dark" otherwise.

    Returns:
        (theta, strength): theta in [0, pi) is the line direction, measured
        from the row axis towards the column axis; strength is >= 0.
    """
    if polarity not in ("bright", "dark"):
        raise ValueError("polarity must be 'bright' or 'dark'")

    planes = np.asarray(planes, dtype=np.float32)
    best_strength = np.full(planes.shape, -np.inf, dtype=np.float32)
    best_theta = np.zeros(planes.shape, dtype=np.float32)

    for sigma in np.atleast_1d(sigmas_px):
        s = (0.0, float(sigma), float(sigma))
        norm = float(sigma) ** 2
        d_rr = norm * ndimage.gaussian_filter(planes, s, order=(0, 2, 0))
        d_cc = norm * ndimage.gaussian_filter(planes, s, order=(0, 0, 2))
        d_rc = norm * ndimage.gaussian_filter(planes, s, order=(0, 1, 1))

        half_trace = 0.5 * (d_rr + d_cc)
        radius = np.sqrt((0.5 * (d_rr - d_cc)) ** 2 + d_rc ** 2)
        # Direction of the largest (most positive) second derivative.
        phi = 0.5 * np.arctan2(2.0 * d_rc, d_rr - d_cc)

        if polarity == "bright":
            strength = radius - half_trace      # minus the smallest eigenvalue
            theta = phi                         # across the line is phi + pi/2
        else:
            strength = radius + half_trace      # the largest eigenvalue
            theta = phi + 0.5 * np.pi           # across the line is phi

        better = strength > best_strength
        best_strength[better] = strength[better]
        best_theta[better] = theta[better]

    return np.mod(best_theta, np.pi).astype(np.float32), np.maximum(best_strength, 0.0)


def _ring_kernels(edges_px: np.ndarray, target_radius_px: float, n_orientations: int):
    """Per-ring pointing, expectation and variance kernels (eqs. 2.2-2.4)."""
    size = int(np.ceil(edges_px[-1]))
    rows, cols = np.mgrid[-size:size + 1, -size:size + 1].astype(np.float64)
    r = np.hypot(rows, cols)
    safe_r = np.where(r > 0, r, np.inf)

    direction = np.mod(np.arctan2(cols, rows), np.pi)
    tolerance = target_radius_px / safe_r
    p = np.minimum(2.0 * target_radius_px / (np.pi * safe_r), 1.0)
    centres = (np.arange(n_orientations) + 0.5) * np.pi / n_orientations

    n_rings = len(edges_px) - 1
    pointing = np.zeros((n_orientations, n_rings) + r.shape, dtype=np.float32)
    expected = np.zeros((n_rings,) + r.shape, dtype=np.float32)
    variance = np.zeros((n_rings,) + r.shape, dtype=np.float32)

    for k in range(n_rings):
        ring = (r >= edges_px[k]) & (r < edges_px[k + 1])
        expected[k] = np.where(ring, p, 0.0)
        variance[k] = np.where(ring, p * (1.0 - p), 0.0)
        for b, centre in enumerate(centres):
            diff = np.abs(direction - centre)
            diff = np.minimum(diff, np.pi - diff)
            pointing[b, k] = ring & (diff < tolerance)

    return pointing, expected, variance, size


def coronal_spiculation_map(volume: np.ndarray,
                            spacing: Sequence[float],
                            depth_axis: int,
                            r_min_mm: float = 3.0,
                            target_radius_mm: float = 3.0,
                            r_max_mm: Sequence[float] = (12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0),
                            sigmas_mm: Sequence[float] = (0.9,),
                            polarity: str = "bright",
                            magnitude_quantile: float = 0.25,
                            n_orientations: int = 36) -> np.ndarray:
    """
    Compute Tan's coronal spiculation value f_m at every voxel.

    Each coronal plane (perpendicular to `depth_axis`) is analysed on its own.
    The neighbourhood sums are separable convolutions, evaluated in the Fourier
    domain: the pointing kernel depends on the line orientation, so voxels are
    grouped into `n_orientations` orientation bins and the bins are summed.

    Tan fixes r_min, R and the r_max range but not the Gaussian scale or the
    strength threshold, so those are parameters here: `sigmas_mm` sets the line
    detector scale and `magnitude_quantile` drops the weakest fraction of
    line responses in the stack.

    Args:
        volume: Intensity volume. Coronal planes must have square pixels, so
            resample anisotropic data first.
        spacing: Voxel size in mm along each axis.
        depth_axis: Axis along the ultrasound beam.
        r_min_mm: Inner radius of the neighbourhood (3 mm in the paper).
        target_radius_mm: Radius R of the central target disk (3 mm).
        r_max_mm: Outer radii to try; f_m is the maximum over them (12-30 mm).
        sigmas_mm: Gaussian scales of the line detector.
        polarity: "bright" (default, white lines) or "dark".
        magnitude_quantile: Fraction of the weakest positive line responses to
            ignore.
        n_orientations: Number of orientation bins.

    Returns:
        float32 array shaped like `volume` holding f_m.
    """
    volume = np.asarray(volume, dtype=np.float32)
    spacing = tuple(float(s) for s in spacing)
    in_plane = [s for axis, s in enumerate(spacing) if axis != depth_axis]
    if not np.isclose(in_plane[0], in_plane[1], rtol=0.05):
        raise ValueError(
            f"coronal planes need square pixels, got in-plane spacing {in_plane}; resample first")
    pixel_mm = float(np.mean(in_plane))

    planes = np.moveaxis(volume, depth_axis, 0)
    theta, strength = line_orientation(planes, [s / pixel_mm for s in sigmas_mm], polarity)

    positive = strength[strength > 0]
    if positive.size == 0:
        return np.zeros_like(volume)
    valid = strength > np.quantile(positive, magnitude_quantile)

    edges = np.array([r_min_mm, *sorted(r_max_mm)], dtype=float) / pixel_mm
    pointing, expected, variance, size = _ring_kernels(edges, target_radius_mm / pixel_mm, n_orientations)
    n_rings = len(edges) - 1

    n_planes, rows, cols = planes.shape
    fr = fft.next_fast_len(rows + size + 1)
    fc = fft.next_fast_len(cols + size + 1)

    def kernel_spectrum(kernel):
        padded = np.zeros((fr, fc), dtype=np.float32)
        padded[:kernel.shape[0], :kernel.shape[1]] = kernel
        return fft.rfft2(np.roll(padded, (-size, -size), axis=(0, 1)))

    def plane_spectrum(stack):
        padded = np.zeros((n_planes, fr, fc), dtype=np.float32)
        padded[:, :rows, :cols] = stack
        return fft.rfft2(padded, axes=(1, 2), workers=-1)

    def spatial(spectrum):
        return fft.irfft2(spectrum, s=(fr, fc), axes=(1, 2), workers=-1)[:, :rows, :cols]

    orientation_bin = np.minimum((theta / np.pi * n_orientations).astype(np.int64), n_orientations - 1)

    pointing_accum = np.zeros((n_rings, n_planes, fr, fc // 2 + 1), dtype=np.complex64)
    for b in range(n_orientations):
        selected = valid & (orientation_bin == b)
        if not selected.any():
            continue
        spectrum = plane_spectrum(selected.astype(np.float32))
        for k in range(n_rings):
            pointing_accum[k] += spectrum * kernel_spectrum(pointing[b, k])[None]

    valid_spectrum = plane_spectrum(valid.astype(np.float32))
    hits = np.cumsum([spatial(pointing_accum[k]) for k in range(n_rings)], axis=0)
    expectation = np.cumsum(
        [spatial(valid_spectrum * kernel_spectrum(expected[k])[None]) for k in range(n_rings)], axis=0)
    spread = np.cumsum(
        [spatial(valid_spectrum * kernel_spectrum(variance[k])[None]) for k in range(n_rings)], axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(spread > 1e-3, (hits - expectation) / np.sqrt(np.maximum(spread, 1e-3)), 0.0)

    return np.moveaxis(f.max(axis=0).astype(np.float32), 0, depth_axis)


def cylinder_spiculation(spiculation: np.ndarray,
                         mask: np.ndarray,
                         spacing: Sequence[float],
                         depth_axis: int,
                         radius_mm: float = 3.0) -> dict:
    """
    Summarise a spiculation map inside Tan et al. (2013)'s central cylinder.

    Spiculation in ABUS peaks along a column through the lesion centre,
    because the pattern repeats in successive coronal planes. The cylinder has
    its axis through the lesion centroid along the depth direction, a 3 mm
    radius, and spans the lesion's depth range.

    Args:
        spiculation: Map from :func:`coronal_spiculation_map`.
        mask: Binary lesion mask of the same shape.
        spacing: Voxel size in mm along each axis.
        depth_axis: Axis along the ultrasound beam; index 0 is the transducer.
        radius_mm: Cylinder radius.

    Returns:
        dict with "mean" (whole cylinder), "upper_mean" (the half nearer the
        skin, above the lesion centre, where Tan found spiculation most
        visible) and "max_slice_mean" (the highest per-coronal-plane mean).
    """
    s = np.moveaxis(np.asarray(spiculation, dtype=np.float32), depth_axis, 0)
    m = np.moveaxis(np.asarray(mask) > 0, depth_axis, 0)
    empty = {"mean": float("nan"), "upper_mean": float("nan"), "max_slice_mean": float("nan")}
    points = np.argwhere(m)
    if points.size == 0:
        return empty

    in_plane = [sp for axis, sp in enumerate(spacing) if axis != depth_axis]
    centre = points[:, 1:].mean(axis=0)
    rr, cc = np.ogrid[:s.shape[1], :s.shape[2]]
    disk = ((rr - centre[0]) * in_plane[0]) ** 2 + ((cc - centre[1]) * in_plane[1]) ** 2 <= radius_mm ** 2
    if not disk.any():
        return empty

    z_min, z_max = int(points[:, 0].min()), int(points[:, 0].max())
    column = s[z_min:z_max + 1][:, disk]
    upper = s[z_min:int(np.floor(points[:, 0].mean())) + 1][:, disk]

    return {
        "mean": float(column.mean()),
        "upper_mean": float(upper.mean()) if upper.size else float("nan"),
        "max_slice_mean": float(column.mean(axis=1).max()),
    }
