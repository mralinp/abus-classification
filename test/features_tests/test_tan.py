import numpy as np
import pytest

from abus_classification.features import tan
from abus_classification.features.tan import lesion as L
from abus_classification.features.tan import spiculation as S


def ball(shape, centre_mm, radii_mm, spacing=(1.0, 1.0, 1.0)):
    """Solid axis-aligned ellipsoid on a grid with the given spacing."""
    grids = np.ogrid[tuple(slice(0, n) for n in shape)]
    total = sum(((g * s - c) / r) ** 2 for g, s, c, r in zip(grids, spacing, centre_mm, radii_mm))
    return total <= 1.0


# --- shape ------------------------------------------------------------------

def test_sphericity_is_one_for_a_sphere_and_lower_for_a_cube():
    sphere = ball((41, 41, 41), (20, 20, 20), (12, 12, 12))
    cube = np.zeros((41, 41, 41), dtype=bool)
    cube[10:31, 10:31, 10:31] = True

    assert L.sphericity(sphere, (1, 1, 1)) > 0.97
    assert L.sphericity(cube, (1, 1, 1)) < L.sphericity(sphere, (1, 1, 1)) - 0.05


def test_sphericity_is_independent_of_voxel_spacing():
    spacing = (0.5, 0.25, 1.0)
    sphere = ball((81, 161, 41), (20, 20, 20), (12, 12, 12), spacing)
    assert L.sphericity(sphere, spacing) > 0.95


def test_compactness_is_near_36_pi_for_a_sphere_and_higher_for_a_cube():
    sphere = ball((61, 61, 61), (30, 30, 30), (20, 20, 20))
    cube = np.zeros((61, 61, 61), dtype=bool)
    cube[15:46, 15:46, 15:46] = True

    # Marching cubes on a voxelised sphere overestimates the area by ~9%
    # (measured 8.5-9.2% for radii 10-40) while the voxel volume is exact, and
    # the cube in A^3 / V^2 turns that into ~29%.
    cp_sphere = L.compactness(sphere, (1, 1, 1))
    assert 36 * np.pi * 1.2 < cp_sphere < 36 * np.pi * 1.4
    assert L.compactness(cube, (1, 1, 1)) > cp_sphere * 1.3


def test_vhwr_of_an_oblate_ellipsoid():
    # depth axis 1: semi-axis 5 in depth, 10 in the coronal plane
    lesion = ball((41, 41, 41), (20, 20, 20), (10, 5, 10))
    assert L.volumetric_height_to_width_ratio(lesion, (1, 1, 1), depth_axis=1) == pytest.approx(11 / 20, rel=0.08)


def test_vhwr_uses_physical_spacing():
    spacing = (0.5, 0.25, 1.0)
    lesion = ball((81, 161, 41), (20, 20, 20), (10, 5, 10), spacing)
    assert L.volumetric_height_to_width_ratio(lesion, spacing, depth_axis=1) == pytest.approx(0.51, rel=0.1)


def test_ellipsoid_fit_recovers_an_ellipsoid():
    lesion = ball((61, 61, 61), (30, 30, 30), (20, 12, 8))
    fit = L.ellipsoid_fit(lesion, (1, 1, 1))
    assert fit["dice"] > 0.95
    assert fit["volume_difference"] / lesion.sum() < 0.05


def test_lesion_volume_in_mm3():
    mask = np.ones((10, 20, 5), dtype=bool)
    assert L.lesion_volume(mask, (0.2, 0.073, 0.475674)) == pytest.approx(1000 * 0.2 * 0.073 * 0.475674)


# --- intensity and margin -----------------------------------------------------

def test_margin_contrast_and_inner_border_variance_of_a_uniform_lesion():
    mask = ball((41, 41, 41), (20, 20, 20), (8, 8, 8))
    volume = np.where(mask, 200.0, 50.0)
    assert L.margin_contrast(volume, mask, (1, 1, 1)) == pytest.approx(150.0)
    assert L.inner_border_variance(volume, mask, (1, 1, 1)) == pytest.approx(0.0)


def test_border_shells_are_1_2_mm_wide():
    mask = np.zeros((1, 41, 1), dtype=bool)
    mask[:, 10:31, :] = True
    inner, outer = L.border_shells(mask, (1.0, 0.6, 1.0), width_mm=1.2)
    assert inner.sum() == 4     # two voxels at each end
    assert outer.sum() == 4


def test_entropy_is_one_bit_for_two_equal_levels():
    mask = np.ones((4, 4, 4), dtype=bool)
    volume = np.zeros((4, 4, 4))
    volume[:2] = 100
    assert L.entropy(volume, mask) == pytest.approx(1.0)


@pytest.mark.parametrize("posterior", [40.0, 160.0])
def test_posterior_acoustic_behavior_measures_shadow_and_enhancement(posterior):
    shape = (41, 60, 41)
    mask = ball(shape, (20, 15, 20), (6, 5, 6))          # lesion spans depth 10-20
    volume = np.full(shape, 100.0)
    footprint = mask.any(axis=1)
    behind = np.zeros(shape, dtype=bool)
    behind[:, 21:40, :] = footprint[:, None, :]
    volume[behind] = posterior

    pab = L.posterior_acoustic_behavior(volume, mask, (1, 1, 1), depth_axis=1)
    assert pab == pytest.approx(posterior - 100.0)


def test_posterior_acoustic_behavior_is_nan_without_tissue_behind():
    mask = np.zeros((20, 10, 20), dtype=bool)
    mask[5:15, 5:10, 5:15] = True
    assert np.isnan(L.posterior_acoustic_behavior(np.ones(mask.shape), mask, (1, 1, 1), depth_axis=1))


# --- spiculation ----------------------------------------------------------------

def test_line_orientation_of_a_bright_line():
    planes = np.zeros((1, 41, 41), dtype=np.float32)
    planes[0, 20, :] = 1.0                                  # runs along the column axis
    theta, strength = S.line_orientation(planes, sigmas_px=(1.5,), polarity="bright")
    assert strength[0, 20, 20] > 10 * strength[0, 10, 20]
    assert abs(theta[0, 20, 20] - np.pi / 2) < 0.05


def star(size=161, n_lines=24, r0=8, r1=55, random_orientation=False, seed=0):
    """Bright radial line segments around the centre, or the same segments at random orientations."""
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size), dtype=np.float32)
    c = size // 2
    for k in range(n_lines):
        a = 2 * np.pi * k / n_lines
        mid = (r0 + r1) / 2 * np.array([np.cos(a), np.sin(a)])
        b = rng.uniform(0, np.pi) if random_orientation else a
        direction = np.array([np.cos(b), np.sin(b)])
        half = (r1 - r0) / 2
        for t in np.linspace(-half, half, int(4 * half)):
            r, col = np.round(c + mid + t * direction).astype(int)
            if 0 <= r < size and 0 <= col < size:
                image[r, col] = 1.0
    return image + 0.05 * rng.standard_normal(image.shape).astype(np.float32)


def test_coronal_spiculation_peaks_at_the_centre_of_a_star():
    stack = lambda plane: np.repeat(plane[:, None, :], 3, axis=1)   # depth axis 1
    spiculated = S.coronal_spiculation_map(stack(star()), (0.6, 0.6, 0.6), depth_axis=1)
    scattered = S.coronal_spiculation_map(stack(star(random_orientation=True)), (0.6, 0.6, 0.6), depth_axis=1)

    centre = spiculated[80, 1, 80]
    assert centre > 5.0
    assert centre >= 0.9 * spiculated[:, 1, :].max()
    assert scattered[80, 1, 80] < centre / 3


def test_coronal_spiculation_requires_square_coronal_pixels():
    with pytest.raises(ValueError):
        S.coronal_spiculation_map(np.zeros((10, 5, 10)), (0.2, 0.073, 0.475674), depth_axis=1)


def test_cylinder_spiculation_summaries():
    spic = np.zeros((21, 11, 21), dtype=np.float32)
    spic[:, :5, :] = 2.0                                     # upper (skin-side) half
    mask = np.zeros_like(spic, dtype=bool)
    mask[5:16, 0:11, 5:16] = True
    out = S.cylinder_spiculation(spic, mask, (1, 1, 1), depth_axis=1, radius_mm=3)
    assert out["upper_mean"] > out["mean"] > 0
    assert out["max_slice_mean"] == pytest.approx(2.0)


# --- end to end -------------------------------------------------------------------

def test_extract_tan_features_on_a_synthetic_dark_lesion():
    spacing = (0.6, 0.6, 0.6)
    shape = (140, 90, 140)
    mask = ball(shape, (42, 20, 42), (6, 5, 6), spacing)
    rng = np.random.default_rng(0)
    volume = (120 + 10 * rng.standard_normal(shape)).astype(np.float32)
    volume[mask] = 60

    features = tan.extract_tan_features(volume, mask, spacing=spacing, depth_axis=1)

    assert set(features) == set(tan.ALL_FEATURES)
    for name in tan.ALL_FEATURES:
        assert np.isfinite(features[name]), name
    assert features["MC"] < 0
    assert features["SH"] > 0.9
    assert features["volume"] == pytest.approx(4 / 3 * np.pi * 6 * 5 * 6, rel=0.1)


def test_extract_keeps_only_the_largest_connected_component():
    spacing = (0.6, 0.6, 0.6)
    shape = (140, 90, 140)
    mask = ball(shape, (42, 20, 42), (6, 5, 6), spacing)
    rng = np.random.default_rng(1)
    volume = (120 + 10 * rng.standard_normal(shape)).astype(np.float32)
    volume[mask] = 60
    speck = np.zeros(shape, dtype=bool)
    speck[3, 85, 3] = True                                   # a stray voxel far from the lesion

    clean = tan.extract_tan_features(volume, mask, spacing=spacing, depth_axis=1)
    noisy = tan.extract_tan_features(volume, mask | speck, spacing=spacing, depth_axis=1)

    for name in tan.ALL_FEATURES:
        assert noisy[name] == pytest.approx(clean[name], rel=1e-6, abs=1e-6), name


def test_feature_sets_match_the_papers():
    assert len(tan.TAN2012_FEATURES) == 11
    assert len(tan.TAN2013_FEATURES) == 14
    assert set(tan.TAN2012_FEATURES) | set(tan.TAN2013_FEATURES) == set(tan.ALL_FEATURES)
