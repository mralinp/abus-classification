"""
Lesion features of Tan et al. for ABUS malignant/benign classification.

Each function follows the definition in Tan's thesis (Radboud University
Nijmegen, 2014): chapter 2 reproduces the 2012 IEEE TMI paper and chapter 3
the 2013 Academic Radiology paper. Equation numbers refer to chapter 2.

All functions take physical voxel spacing in mm, so they work on the
anisotropic TDSC-ABUS volumes directly as well as on resampled ones.
"""
from typing import Sequence, Tuple

import numpy as np
from scipy import ndimage
from skimage import measure


def _as_mask(mask) -> np.ndarray:
    return np.asarray(mask) > 0


def _window(mask: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """`mask` sampled over the index box [lo, hi), which may extend past the array; outside is False."""
    out = np.zeros(tuple(int(v) for v in hi - lo), dtype=bool)
    src_lo = np.maximum(lo, 0)
    src_hi = np.minimum(hi, mask.shape)
    if np.any(src_hi <= src_lo):
        return out
    dst_lo = src_lo - lo
    dst = tuple(slice(int(d), int(d + (b - a))) for d, a, b in zip(dst_lo, src_lo, src_hi))
    src = tuple(slice(int(a), int(b)) for a, b in zip(src_lo, src_hi))
    out[dst] = mask[src]
    return out


def _axis_offsets_mm(lo, hi, centre_mm, spacing) -> list:
    return [np.arange(l, h) * s - c for l, h, c, s in zip(lo, hi, centre_mm, spacing)]


# --- echotexture and echogenicity ----------------------------------------

def average_intensity(volume: np.ndarray, mask: np.ndarray) -> float:
    """Average intensity (AI) of the voxels inside the lesion; also Tan 2013's echogenicity."""
    values = np.asarray(volume)[_as_mask(mask)]
    return float(values.mean()) if values.size else float("nan")


def variance_of_intensities(volume: np.ndarray, mask: np.ndarray) -> float:
    """Variance of intensities (VI) of the voxels inside the lesion."""
    values = np.asarray(volume)[_as_mask(mask)]
    return float(values.var()) if values.size else float("nan")


def entropy(volume: np.ndarray, mask: np.ndarray, bins: int = 256,
            value_range: Tuple[float, float] = (0.0, 256.0)) -> float:
    """
    Entropy (H) of the lesion's intensity histogram, in bits (eq. 2.6).

        H = -sum_i p_i log2(p_i)

    where p_i is the normalised count of histogram bin i over lesion voxels.
    """
    values = np.asarray(volume)[_as_mask(mask)]
    if values.size == 0:
        return float("nan")
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


# --- margin ---------------------------------------------------------------

def border_shells(mask: np.ndarray, spacing: Sequence[float],
                  width_mm: float = 1.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inner and outer border of a lesion.

    The inner border is lesion voxels within `width_mm` of the boundary, the
    outer border non-lesion voxels within `width_mm` of it. Tan uses 1.2 mm,
    two voxels at the 0.6 mm resolution the system runs on.
    """
    mask = _as_mask(mask)
    if not mask.any():
        empty = np.zeros_like(mask)
        return empty, empty
    spacing = tuple(float(s) for s in spacing)
    inner = mask & (ndimage.distance_transform_edt(mask, sampling=spacing) <= width_mm)
    if mask.all():
        return inner, np.zeros_like(mask)
    outer = ~mask & (ndimage.distance_transform_edt(~mask, sampling=spacing) <= width_mm)
    return inner, outer


def margin_contrast(volume: np.ndarray, mask: np.ndarray, spacing: Sequence[float],
                    width_mm: float = 1.2) -> float:
    """
    Margin contrast (MC): mean intensity of the inner border minus the outer border.

    Benign lesions usually have a sharp demarcation from surrounding tissue,
    malignant ones an indistinct margin. Returns nan when either border is
    empty, e.g. when the array holds no tissue around the lesion.
    """
    volume = np.asarray(volume)
    inner, outer = border_shells(mask, spacing, width_mm)
    if not inner.any() or not outer.any():
        return float("nan")
    return float(volume[inner].mean() - volume[outer].mean())


def inner_border_variance(volume: np.ndarray, mask: np.ndarray, spacing: Sequence[float],
                          width_mm: float = 1.2) -> float:
    """Intensity variance of the inner border (voxels within 1.2 mm of the boundary), from Tan 2013."""
    volume = np.asarray(volume)
    inner, _ = border_shells(mask, spacing, width_mm)
    return float(volume[inner].var()) if inner.any() else float("nan")


# --- shape ----------------------------------------------------------------

def lesion_volume(mask: np.ndarray, spacing: Sequence[float]) -> float:
    """Lesion volume in mm^3 (Tan 2013)."""
    return float(_as_mask(mask).sum() * np.prod(np.asarray(spacing, dtype=float)))


def surface_area(mask: np.ndarray, spacing: Sequence[float]) -> float:
    """Area in mm^2 of the lesion's marching-cubes surface; the mask is padded so the surface is closed."""
    mask = _as_mask(mask)
    if not mask.any():
        return float("nan")
    verts, faces, _, _ = measure.marching_cubes(
        np.pad(mask, 1).astype(np.uint8), level=0.5, spacing=tuple(float(s) for s in spacing))
    return float(measure.mesh_surface_area(verts, faces))


def volumetric_height_to_width_ratio(mask: np.ndarray, spacing: Sequence[float],
                                     depth_axis: int) -> float:
    """
    Volumetric height-to-width ratio (VHWR), eq. 2.7.

        VHWR = h / w

    h is the lesion's extent along the depth direction. w is the effective
    diameter of the coronal cross-section through the lesion centre: the
    diameter of the circle with the same area. Lesions wider than they are
    tall are more likely benign.

    If a concave lesion does not cover its own centroid plane, the largest
    coronal cross-section is used instead.
    """
    spacing = np.asarray(spacing, dtype=float)
    m = np.moveaxis(_as_mask(mask), depth_axis, 0)
    points = np.argwhere(m)
    if points.size == 0:
        return float("nan")

    height = (points[:, 0].max() - points[:, 0].min() + 1) * spacing[depth_axis]
    pixel_area = float(np.prod(np.delete(spacing, depth_axis)))
    area = m[int(round(points[:, 0].mean()))].sum() * pixel_area
    if area == 0:
        area = m.sum(axis=(1, 2)).max() * pixel_area
    width = 2.0 * np.sqrt(area / np.pi)
    return float(height / width)


def sphericity(mask: np.ndarray, spacing: Sequence[float]) -> float:
    """
    Sphericity (SH), eq. 2.8.

        SH = |V_l intersect S| / |S|

    where S is a sphere at the lesion centre with the same volume as the
    lesion V_l. It is 1 when the lesion is a sphere and falls as the shape
    departs from one. Note this is Tan's overlap-based definition, not the
    surface-area based sphericity.
    """
    mask = _as_mask(mask)
    spacing = np.asarray(spacing, dtype=float)
    points = np.argwhere(mask)
    if points.size == 0:
        return float("nan")

    volume = points.shape[0] * float(np.prod(spacing))
    radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    centre = points.mean(axis=0) * spacing

    lo = np.floor((centre - radius) / spacing).astype(int) - 1
    hi = np.ceil((centre + radius) / spacing).astype(int) + 2
    d0, d1, d2 = _axis_offsets_mm(lo, hi, centre, spacing)
    inside = (d0[:, None, None] ** 2 + d1[None, :, None] ** 2 + d2[None, None, :] ** 2) <= radius ** 2

    n_sphere = int(inside.sum())
    if n_sphere == 0:
        return float("nan")
    return float((inside & _window(mask, lo, hi)).sum() / n_sphere)


def compactness(mask: np.ndarray, spacing: Sequence[float]) -> float:
    """
    Compactness (CP), eq. 2.9.

        CP = A^3 / |V_l|^2

    with A the lesion surface area. The minimum, 36 pi, is reached by a sphere;
    irregular, lobulated lesions score higher.
    """
    volume = lesion_volume(mask, spacing)
    if volume == 0:
        return float("nan")
    return float(surface_area(mask, spacing) ** 3 / volume ** 2)


def ellipsoid_fit(mask: np.ndarray, spacing: Sequence[float]) -> dict:
    """
    Fit an ellipsoid to the lesion and compare the two, after Tan 2013.

    Tan registers a sphere to the lesion with an affine transform and reports
    the Dice coefficient and the volume difference (DF) between the result and
    the lesion. An affinely transformed sphere is an ellipsoid; here it is the
    ellipsoid with the lesion's centroid and second moments (a solid ellipsoid
    with semi-axis a has variance a^2 / 5 along it). That is a closed-form
    stand-in for the paper's iterative registration, not an exact replica.

    Returns:
        dict with "dice" and "volume_difference" (mm^3, absolute).
    """
    mask = _as_mask(mask)
    spacing = np.asarray(spacing, dtype=float)
    points = np.argwhere(mask)
    if points.shape[0] < 4:
        return {"dice": float("nan"), "volume_difference": float("nan")}

    coords = points * spacing
    centre = coords.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(coords, rowvar=False))
    semi_axes = np.sqrt(5.0 * np.maximum(eigvals, 1e-12))
    half_extent = np.sqrt((eigvecs ** 2 * semi_axes ** 2).sum(axis=1))

    lo = np.floor((centre - half_extent) / spacing).astype(int) - 1
    hi = np.ceil((centre + half_extent) / spacing).astype(int) + 2
    grid = np.stack(np.meshgrid(*_axis_offsets_mm(lo, hi, centre, spacing), indexing="ij"), axis=-1)
    inside = (((grid @ eigvecs) / semi_axes) ** 2).sum(axis=-1) <= 1.0

    n_ellipsoid, n_lesion = int(inside.sum()), points.shape[0]
    overlap = int((inside & _window(mask, lo, hi)).sum())
    return {
        "dice": 2.0 * overlap / (n_ellipsoid + n_lesion),
        "volume_difference": abs(n_ellipsoid - n_lesion) * float(np.prod(spacing)),
    }


# --- posterior acoustic behavior -----------------------------------------

def posterior_acoustic_behavior(volume: np.ndarray, mask: np.ndarray, spacing: Sequence[float],
                                depth_axis: int, ring_mm: float = 2.0) -> float:
    """
    Posterior acoustic behavior (PAB), eqs. 2.10-2.12.

        PAB = mean(I in V_p) - mean(I in V_s)

    With d(x, y) the signed distance map of the lesion's 2D coronal projection
    (negative inside), and z_max the lesion's deepest plane:

        V_p = {d < 0,          z_max < z <= z_max + l}   posterior region
        V_s = {0 < d <= ring,  z_max < z <= z_max + l}   surrounding region

    l equals the lesion height. Negative values mean shadowing (typical of
    malignancy), positive values enhancement (typical of cysts).

    The thesis writes d <= 2 without a unit; it is taken here as 2 mm.

    Returns:
        nan when no tissue lies behind the lesion or around its footprint.
    """
    volume = np.asarray(volume, dtype=np.float32)
    mask = _as_mask(mask)
    if volume.shape != mask.shape:
        raise ValueError("volume and mask must have the same shape")
    if not mask.any():
        return float("nan")

    spacing = np.asarray(spacing, dtype=float)
    vol = np.moveaxis(volume, depth_axis, 0)
    m = np.moveaxis(mask, depth_axis, 0)
    in_plane = tuple(np.delete(spacing, depth_axis))

    footprint = m.any(axis=0)
    signed = np.where(footprint,
                      -ndimage.distance_transform_edt(footprint, sampling=in_plane),
                      ndimage.distance_transform_edt(~footprint, sampling=in_plane))
    posterior = signed < 0
    surround = (signed > 0) & (signed <= ring_mm)

    depths = np.flatnonzero(m.any(axis=(1, 2)))
    z_max = int(depths[-1])
    height = z_max - int(depths[0]) + 1
    start, stop = z_max + 1, min(z_max + 1 + height, vol.shape[0])
    if start >= stop or not surround.any():
        return float("nan")

    slab = vol[start:stop]
    return float(slab[:, posterior].mean() - slab[:, surround].mean())
