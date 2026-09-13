"""Feature sets of Tan et al. (2012, 2013) for ABUS lesion classification."""
from typing import Optional, Sequence, Tuple

import numpy as np

from abus_classification.features.tan import lesion as L
from abus_classification.features.tan.spiculation import coronal_spiculation_map, cylinder_spiculation
from abus_classification.utils.spacing import TDSC_DEPTH_AXIS, TDSC_SPACING, prepare_lesion

#: The 11 features of Tan et al., IEEE TMI 2012 (thesis table 2.2).
TAN2012_FEATURES: Tuple[str, ...] = (
    "SPmean", "SP90", "SP75", "VI", "H", "AI", "MC", "VHWR", "SH", "CP", "PAB",
)

#: The 14 features of Tan et al., Academic Radiology 2013 (thesis table 3.2).
TAN2013_FEATURES: Tuple[str, ...] = (
    "SPmean", "VHWR", "MC", "PAB", "H", "DICE", "echogenicity",
    "SPcyl_mean", "SPcyl_upper_mean", "SPcyl_max_slice_mean",
    "VI", "DF", "IBV", "volume",
)

#: Every feature computed by :func:`extract_tan_features`.
ALL_FEATURES: Tuple[str, ...] = tuple(dict.fromkeys(TAN2012_FEATURES + TAN2013_FEATURES))

FEATURE_DESCRIPTIONS = {
    "SPmean": "mean coronal spiculation inside the lesion",
    "SP90": "90th percentile of coronal spiculation inside the lesion",
    "SP75": "75th percentile of coronal spiculation inside the lesion",
    "VI": "variance of intensities inside the lesion",
    "H": "entropy of the lesion intensity histogram (bits)",
    "AI": "average intensity inside the lesion",
    "echogenicity": "average intensity inside the lesion (Tan 2013 name for AI)",
    "MC": "margin contrast: inner minus outer 1.2 mm border intensity",
    "VHWR": "volumetric height-to-width ratio",
    "SH": "overlap with an equal-volume sphere at the lesion centre",
    "CP": "compactness, surface area^3 / volume^2",
    "PAB": "posterior acoustic behavior: posterior minus surrounding intensity",
    "DICE": "Dice overlap with the fitted ellipsoid",
    "DF": "volume difference from the fitted ellipsoid (mm^3)",
    "SPcyl_mean": "mean spiculation in a 3 mm central cylinder along depth",
    "SPcyl_upper_mean": "mean spiculation in the upper (skin-side) half of the cylinder",
    "SPcyl_max_slice_mean": "highest per-coronal-plane mean spiculation in the cylinder",
    "IBV": "intensity variance of the inner 1.2 mm border",
    "volume": "lesion volume (mm^3)",
}


def extract_tan_features(volume: np.ndarray,
                         mask: np.ndarray,
                         spacing: Sequence[float] = TDSC_SPACING,
                         depth_axis: int = TDSC_DEPTH_AXIS,
                         target_mm: float = 0.6,
                         spiculation_kwargs: Optional[dict] = None,
                         return_maps: bool = False):
    """
    Compute every Tan et al. (2012, 2013) feature for one lesion.

    Args:
        volume: Full intensity volume.
        mask: Binary lesion mask of the same shape.
        spacing: Voxel size in mm along each axis. Defaults to TDSC-ABUS.
        depth_axis: Axis along the ultrasound beam. Defaults to TDSC-ABUS.
        target_mm: Isotropic voxel size the features are computed at.
        spiculation_kwargs: Extra arguments for :func:`coronal_spiculation_map`.
        return_maps: Also return the resampled crop and its spiculation map.

    Returns:
        dict mapping each name in :data:`ALL_FEATURES` to a float. Select
        :data:`TAN2012_FEATURES` or :data:`TAN2013_FEATURES` for either paper.
        With `return_maps`, a second dict holds "volume", "mask", "spiculation"
        (restricted to the lesion's depth range) and "spacing".
    """
    vol, m, sp = prepare_lesion(volume, mask, spacing, depth_axis, target_mm)
    if not m.any():
        nan = {name: float("nan") for name in ALL_FEATURES}
        return (nan, {}) if return_maps else nan

    # Spiculation is only summarised over the lesion's depth range, so the map
    # is computed on those coronal planes alone.
    depths = np.flatnonzero(np.moveaxis(m, depth_axis, 0).any(axis=(1, 2)))
    planes = [slice(None)] * m.ndim
    planes[depth_axis] = slice(int(depths[0]), int(depths[-1]) + 1)
    planes = tuple(planes)

    spic = coronal_spiculation_map(vol[planes], sp, depth_axis, **(spiculation_kwargs or {}))
    lesion_spic = spic[m[planes]]
    cylinder = cylinder_spiculation(spic, m[planes], sp, depth_axis)
    ellipsoid = L.ellipsoid_fit(m, sp)
    ai = L.average_intensity(vol, m)

    features = {
        "SPmean": float(lesion_spic.mean()),
        "SP90": float(np.percentile(lesion_spic, 90)),
        "SP75": float(np.percentile(lesion_spic, 75)),
        "VI": L.variance_of_intensities(vol, m),
        "H": L.entropy(vol, m),
        "AI": ai,
        "echogenicity": ai,
        "MC": L.margin_contrast(vol, m, sp),
        "VHWR": L.volumetric_height_to_width_ratio(m, sp, depth_axis),
        "SH": L.sphericity(m, sp),
        "CP": L.compactness(m, sp),
        "PAB": L.posterior_acoustic_behavior(vol, m, sp, depth_axis),
        "DICE": ellipsoid["dice"],
        "DF": ellipsoid["volume_difference"],
        "SPcyl_mean": cylinder["mean"],
        "SPcyl_upper_mean": cylinder["upper_mean"],
        "SPcyl_max_slice_mean": cylinder["max_slice_mean"],
        "IBV": L.inner_border_variance(vol, m, sp),
        "volume": L.lesion_volume(m, sp),
    }
    features = {name: features[name] for name in ALL_FEATURES}

    if return_maps:
        return features, {"volume": vol, "mask": m, "spiculation": spic, "planes": planes, "spacing": sp}
    return features
