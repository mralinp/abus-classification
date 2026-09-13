import numpy as np
import pytest

from abus_classification.features import radiology
from abus_classification.utils.spacing import resample_mask_to_spacing


def ball(shape, centre_mm, radii_mm, spacing=(1.0, 1.0, 1.0)):
    """Solid axis-aligned ellipsoid on a grid with the given spacing."""
    grids = np.ogrid[tuple(slice(0, n) for n in shape)]
    total = sum(((g * s - c) / r) ** 2 for g, s, c, r in zip(grids, spacing, centre_mm, radii_mm))
    return total <= 1.0


ANISOTROPIC = (0.5, 0.25, 1.0)


def test_sphericity_and_compactness_need_physical_spacing():
    iso = ball((61, 61, 61), (30, 30, 30), (20, 20, 20))
    aniso = ball((121, 241, 61), (30, 30, 30), (20, 20, 20), ANISOTROPIC)
    resampled, spacing = resample_mask_to_spacing(aniso, ANISOTROPIC, 1.0)

    for feature in (radiology.sphericity, radiology.compactness):
        reference = feature(iso)
        in_voxels = feature(aniso)
        with_spacing = feature(aniso, ANISOTROPIC)
        # resampling to cubic voxels recovers the value measured on cubic voxels
        assert feature(resampled, spacing) == pytest.approx(reference, rel=0.03), feature.__name__
        # spacing alone removes most of the error; marching cubes still overestimates
        # area on an anisotropic staircase, which is why the extractor resamples
        assert abs(with_spacing - reference) < abs(in_voxels - reference) / 2, feature.__name__


def test_height_to_width_ratio_uses_depth_over_transducer_width():
    # radii: 10 mm along the transducer (axis 0), 5 mm in depth (axis 1), 20 mm along the sweep (axis 2)
    iso = ball((41, 21, 61), (20, 10, 30), (10, 5, 20))
    assert radiology.height_to_width_ratio(iso) == pytest.approx(11 / 21, rel=0.05)

    aniso = ball((81, 81, 61), (20, 10, 30), (10, 5, 20), ANISOTROPIC)
    assert radiology.height_to_width_ratio(aniso, ANISOTROPIC) == pytest.approx(0.5, rel=0.06)


def test_elongation_and_flatness_are_independent_of_voxel_spacing():
    aniso = ball((81, 161, 61), (20, 20, 30), (20, 10, 5), ANISOTROPIC)
    assert radiology.elongation(aniso, ANISOTROPIC) == pytest.approx(0.5, abs=0.03)
    assert radiology.flatness(aniso, ANISOTROPIC) == pytest.approx(0.25, abs=0.03)


def test_margin_contrast_with_physical_shells():
    mask = ball((81, 161, 41), (20, 20, 20), (8, 8, 8), ANISOTROPIC)
    volume = np.where(mask, 200.0, 50.0)
    assert radiology.margin_contrast(volume, mask, spacing=ANISOTROPIC, width_mm=1.2) == pytest.approx(150.0)
    assert radiology.margin_contrast(volume, mask) == pytest.approx(150.0)   # one-voxel shells still work


def test_extract_radiology_features_on_a_synthetic_dark_lesion():
    spacing = (0.6, 0.6, 0.6)
    shape = (120, 90, 120)
    mask = ball(shape, (36, 20, 36), (8, 5, 6), spacing)
    rng = np.random.default_rng(0)
    volume = np.clip(120 + 10 * rng.standard_normal(shape), 0, 255).astype(np.float32)
    volume[mask] = 60

    features = radiology.extract_radiology_features(volume, mask, spacing=spacing, depth_axis=1)

    assert tuple(features) == radiology.RADIOLOGY_FEATURES
    for name, value in features.items():
        assert np.isfinite(value), name
    assert features["volume"] == pytest.approx(4 / 3 * np.pi * 8 * 5 * 6, rel=0.1)
    assert features["height_to_width_ratio"] == pytest.approx(10 / 16, rel=0.15)
    assert features["margin_contrast"] < 0
