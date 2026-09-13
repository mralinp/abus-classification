import numpy as np


def height_to_width_ratio(mask: np.ndarray, spacing=None, depth_axis: int = 1, width_axis: int = 0) -> float:
    """
    Calculate the height-to-width ratio of a 3D lesion mask.

    Height is the lesion's extent along the ultrasound beam (`depth_axis`) and
    width its extent along the transducer (`width_axis`), both in physical
    units when `spacing` is given. A ratio above 1 is the "taller-than-wide"
    sign associated with malignancy in breast ultrasound, which radiologists
    judge on the acquired B-mode image.

    For TDSC-ABUS that image spans axes 0 and 1: its pixels measure 0.200 mm
    along the transducer (axis 0) and 0.073 mm in depth (axis 1), and the
    images are stacked 0.476 mm apart along axis 2. Hence the defaults. Pass
    the spacing: in voxel units the depth extent is inflated 2.7x relative to
    the width.

    Args:
        mask (np.ndarray): A 3D numpy array containing the mask of the lesion.
        spacing (tuple, optional): Physical voxel size along each axis.
        depth_axis (int): Axis along the ultrasound beam.
        width_axis (int): Axis along the transducer.

    Returns:
        float: height / width, or nan if the mask is empty.
    """
    mask = np.asarray(mask) > 0
    assert mask.ndim == 3, "The mask should be a 3D ndarray."
    if depth_axis == width_axis:
        raise ValueError("depth_axis and width_axis must differ")

    points = np.argwhere(mask)
    if points.size == 0:
        return float("nan")

    spacing = np.ones(3) if spacing is None else np.asarray(spacing, dtype=float)
    extents = (np.ptp(points, axis=0) + 1) * spacing
    return float(extents[depth_axis] / extents[width_axis])
