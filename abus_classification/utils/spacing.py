"""Voxel spacing helpers for anisotropic ABUS volumes."""
from typing import Sequence, Tuple

import numpy as np
from scipy import ndimage

#: Voxel spacing of the TDSC-ABUS 2023 volumes, in mm, along array axes (0, 1, 2).
#:
#: The NRRD files carry identity ``space directions``, so the spacing has to be
#: supplied. The TDSC-ABUS challenge paper reports 0.200 mm and 0.073 mm pixel
#: spacing and 0.475674 mm between slices. Axis 1 is the ultrasound depth
#: direction: mean intensity falls steadily along it, and 608 voxels at
#: 0.073 mm give the ~44 mm scan depth of an ABUS acquisition.
TDSC_SPACING: Tuple[float, float, float] = (0.2, 0.073, 0.475674)

#: Array axis of the TDSC-ABUS volumes that runs along the ultrasound beam.
#: Slices perpendicular to it are coronal planes, parallel to the skin, with
#: index 0 at the transducer.
TDSC_DEPTH_AXIS: int = 1


def crop_around_mask(mask: np.ndarray,
                     spacing: Sequence[float],
                     margin_before_mm,
                     margin_after_mm=None) -> Tuple[slice, ...]:
    """
    Bounding-box slices of a mask, grown by a physical margin and clipped to the array.

    Args:
        mask: Binary array.
        spacing: Voxel size in mm along each axis.
        margin_before_mm: Margin added before the lesion along each axis; one
            value for every axis or one per axis.
        margin_after_mm: Margin added after the lesion. Defaults to
            `margin_before_mm`.

    Returns:
        A tuple of slices, one per axis.
    """
    mask = np.asarray(mask) > 0
    points = np.argwhere(mask)
    if points.size == 0:
        raise ValueError("mask is empty")

    spacing = np.asarray(spacing, dtype=float)
    before = np.broadcast_to(np.asarray(margin_before_mm, dtype=float), spacing.shape)
    after = before if margin_after_mm is None else np.broadcast_to(
        np.asarray(margin_after_mm, dtype=float), spacing.shape)

    lo = np.maximum(points.min(axis=0) - np.ceil(before / spacing).astype(int), 0)
    hi = np.minimum(points.max(axis=0) + np.ceil(after / spacing).astype(int) + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def resample_to_spacing(array: np.ndarray,
                        spacing: Sequence[float],
                        target_mm: float = 0.6,
                        order: int = 1) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """
    Resample an array to isotropic voxels of roughly `target_mm`.

    Args:
        array: Array to resample.
        spacing: Voxel size in mm along each axis.
        target_mm: Desired voxel size.
        order: Spline interpolation order (1 = linear).

    Returns:
        The resampled float32 array and its actual spacing per axis. The output
        shape is the input shape scaled by spacing / target_mm and rounded, so
        the actual spacing differs slightly from `target_mm`.
    """
    array = np.asarray(array)
    spacing = np.asarray(spacing, dtype=float)
    zoom = spacing / float(target_mm)
    out = ndimage.zoom(array.astype(np.float32), zoom, order=order, mode="nearest", grid_mode=True)
    actual = spacing * np.asarray(array.shape) / np.asarray(out.shape)
    return out, tuple(float(s) for s in actual)


def resample_mask_to_spacing(mask: np.ndarray,
                             spacing: Sequence[float],
                             target_mm: float = 0.6) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """
    Resample a binary mask to isotropic voxels of roughly `target_mm`.

    Uses linear interpolation followed by a 0.5 threshold, which preserves the
    lesion volume far better than nearest-neighbour resampling.
    """
    out, actual = resample_to_spacing((np.asarray(mask) > 0).astype(np.float32), spacing, target_mm, order=1)
    return out > 0.5, actual


def prepare_lesion(volume: np.ndarray,
                   mask: np.ndarray,
                   spacing: Sequence[float] = TDSC_SPACING,
                   depth_axis: int = TDSC_DEPTH_AXIS,
                   target_mm: float = 0.6,
                   lateral_margin_mm: float = 32.0,
                   border_margin_mm: float = 3.0,
                   largest_component: bool = True):
    """
    Crop a lesion with the context every feature needs and resample it to 0.6 mm cubic voxels.

    The margins cover the 30 mm spiculation neighbourhood in the coronal plane,
    the 1.2 mm border shells, and the posterior region, which reaches one
    lesion height below the lesion. Tan et al. (2013) run their system on
    0.6 mm cubic voxels, and texture features need square pixels.

    Args:
        volume: Full intensity volume (not a tight bounding-box crop, or the
            spiculation and posterior-acoustic regions fall outside it).
        mask: Binary lesion mask of the same shape.
        spacing: Voxel size in mm along each axis.
        depth_axis: Axis along the ultrasound beam.
        target_mm: Output voxel size.
        lateral_margin_mm: Context kept around the lesion in the coronal plane.
        border_margin_mm: Context kept above the lesion, and below it in
            addition to the posterior region.
        largest_component: Keep only the largest connected region of the mask.
            Tan's segmentation always yields one region, while 9 of the 200
            TDSC-ABUS masks carry extra fragments, mostly specks of a few voxels
            that would otherwise shift the centroid, the coronal footprint and
            the surface area.

    Returns:
        (volume, mask, spacing) of the resampled crop.
    """
    mask = np.asarray(mask) > 0
    spacing = np.asarray(spacing, dtype=float)
    points = np.argwhere(mask)
    if points.size == 0:
        raise ValueError("mask is empty")

    if largest_component:
        lo, hi = points.min(axis=0), points.max(axis=0) + 1
        tight = mask[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]
        labels, n = ndimage.label(tight, structure=np.ones((3,) * mask.ndim))
        if n > 1:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            points = np.argwhere(labels == sizes.argmax()) + lo

    height_mm = (np.ptp(points[:, depth_axis]) + 1) * spacing[depth_axis]
    before = np.full(mask.ndim, lateral_margin_mm)
    after = np.full(mask.ndim, lateral_margin_mm)
    before[depth_axis] = border_margin_mm
    after[depth_axis] = height_mm + border_margin_mm

    lo = np.maximum(points.min(axis=0) - np.ceil(before / spacing).astype(int), 0)
    hi = np.minimum(points.max(axis=0) + np.ceil(after / spacing).astype(int) + 1, mask.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

    crop_mask = np.zeros(tuple(int(v) for v in hi - lo), dtype=bool)
    crop_mask[tuple((points - lo).T)] = True

    vol_r, spacing_r = resample_to_spacing(np.asarray(volume)[box], spacing, target_mm)
    mask_r, _ = resample_mask_to_spacing(crop_mask, spacing, target_mm)
    return vol_r, mask_r, spacing_r
