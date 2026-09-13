"""A general radiomic feature set for ABUS lesions, in physical units."""
from typing import Sequence

import numpy as np

from abus_classification.features.radiology.average_intensity import average_intensity
from abus_classification.features.radiology.compactness import compactness
from abus_classification.features.radiology.elongation import bounding_box_fill, elongation, flatness
from abus_classification.features.radiology.entropy import entropy
from abus_classification.features.radiology.margin_contrast import margin_contrast
from abus_classification.features.radiology.pab import posterior_acoustic_behavior
from abus_classification.features.radiology.sphericity import sphericity
from abus_classification.features.radiology.spiculation import spiculation
from abus_classification.features.radiology.surface_area import surface_area, surface_to_volume_ratio
from abus_classification.features.radiology.variance_of_intensities import variance_of_intensities
from abus_classification.features.radiology.vhwr import height_to_width_ratio
from abus_classification.features.radiology.volume import lesion_volume
from abus_classification.features.texture.glcm import glcm
from abus_classification.utils.spacing import TDSC_DEPTH_AXIS, TDSC_SPACING, prepare_lesion

#: Feature names by group, in output order.
FEATURE_GROUPS = {
    "shape": ("volume", "surface_area", "surface_to_volume", "sphericity", "compactness",
              "height_to_width_ratio", "elongation", "flatness", "bounding_box_fill"),
    "margin": ("margin_contrast", "boundary_fractal_dimension", "radial_distance_deviation"),
    "echo pattern": ("mean_intensity", "intensity_variance", "entropy", "posterior_acoustic"),
    "texture": ("glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy", "glcm_correlation"),
}

#: Every feature computed by :func:`extract_radiology_features`.
RADIOLOGY_FEATURES = tuple(name for names in FEATURE_GROUPS.values() for name in names)

_GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")


def extract_radiology_features(volume: np.ndarray,
                               mask: np.ndarray,
                               spacing: Sequence[float] = TDSC_SPACING,
                               depth_axis: int = TDSC_DEPTH_AXIS,
                               width_axis: int = 0,
                               target_mm: float = 0.6) -> dict:
    """
    Compute the general radiomic features of one lesion, in physical units.

    The lesion is cropped with a few millimetres of surrounding tissue and a
    posterior region, and resampled to `target_mm` cubic voxels, so shape
    features see true proportions and texture is measured on square pixels.

    Texture is the grey-level co-occurrence matrix of the lesion's widest
    coronal cross-section, cropped to the lesion, at a one-voxel distance and
    averaged over four directions.

    Args:
        volume: Full intensity volume.
        mask: Binary lesion mask of the same shape.
        spacing: Voxel size in mm along each axis. Defaults to TDSC-ABUS.
        depth_axis: Axis along the ultrasound beam. Defaults to TDSC-ABUS.
        width_axis: Axis along the transducer, for the height-to-width ratio.
        target_mm: Isotropic voxel size the features are computed at.

    Returns:
        dict mapping each name in :data:`RADIOLOGY_FEATURES` to a float.
        Volume is in mm^3 and surface area in mm^2.
    """
    vol, m, sp = prepare_lesion(volume, mask, spacing, depth_axis, target_mm,
                                lateral_margin_mm=5.0, border_margin_mm=3.0)
    if not m.any():
        return {name: float("nan") for name in RADIOLOGY_FEATURES}

    margin = spiculation(m)

    planes = np.moveaxis(vol, depth_axis, 0)
    plane_mask = np.moveaxis(m, depth_axis, 0)
    widest = int(plane_mask.sum(axis=(1, 2)).argmax())
    rows, cols = np.nonzero(plane_mask[widest])
    patch = planes[widest, rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    matrix = glcm(np.clip(np.round(patch), 0, 255).astype(np.uint8), distances=[1],
                  angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])

    depth_voxel = sp[depth_axis]
    values = {
        "volume": lesion_volume(m, sp),
        "surface_area": surface_area(m, sp),
        "surface_to_volume": surface_to_volume_ratio(m, sp),
        "sphericity": sphericity(m, sp),
        "compactness": compactness(m, sp),
        "height_to_width_ratio": height_to_width_ratio(m, sp, depth_axis=depth_axis, width_axis=width_axis),
        "elongation": elongation(m, sp),
        "flatness": flatness(m, sp),
        "bounding_box_fill": bounding_box_fill(m),
        "margin_contrast": margin_contrast(vol, m, spacing=sp, width_mm=1.2),
        "boundary_fractal_dimension": margin["boundary_fractal_dimension"],
        "radial_distance_deviation": margin["radial_distance_deviation"],
        "mean_intensity": average_intensity(vol, m),
        "intensity_variance": variance_of_intensities(vol, m),
        "entropy": entropy(vol, m),
        "posterior_acoustic": posterior_acoustic_behavior(
            vol, m, axis=depth_axis,
            margin=max(1, int(round(2.0 / depth_voxel))), depth=max(1, int(round(12.0 / depth_voxel)))),
        **{f"glcm_{name}": float(np.mean(matrix[name])) for name in _GLCM_PROPERTIES},
    }
    return {name: float(values[name]) for name in RADIOLOGY_FEATURES}
