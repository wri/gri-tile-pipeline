"""Post-processing: bright surface removal, nodata masking.

Ported from reference ``download_and_predict_job.py`` lines 1216-1239.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt as distance


def identify_bright_surfaces(s2_median: np.ndarray) -> np.ndarray:
    """Identify bright bare surfaces to attenuate false-positive tree cover.

    Bright surfaces are pixels where:
      - NIR/SWIR ratio < 0.9
      - Mean visible reflectance > 0.2
      - EVI < 0.3

    Args:
        s2_median: ``(T, H, W, B)`` Sentinel-2 reflectance array (10 bands).
            Typically the temporal median or a representative composite.

    Returns:
        ``(H_inner, W_inner)`` float32 attenuation mask in [0, 1].
        0 = suppress prediction, 1 = keep prediction.
        Inner dimensions exclude a 7-pixel border (``[7:-7, 7:-7]``).
    """
    # EVI inline
    blue = np.clip(s2_median[..., 0], 0, 1)
    red = np.clip(s2_median[..., 2], 0, 1)
    nir = np.clip(s2_median[..., 3], 0, 1)
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    evi = np.clip(evi, -1.5, 1.5)

    swir16 = np.clip(s2_median[..., 8], 0, 1)
    nir_swir_ratio = nir / (swir16 + 0.01)
    bright = nir_swir_ratio < 0.9
    bright = bright * (np.mean(s2_median[..., :3], axis=-1) > 0.2)
    bright = bright * (evi < 0.3)

    # Aggregate over time: bright in more than 1 timestep
    bright_surface = np.sum(bright, axis=0) > 1

    # Morphological cleanup
    bright_surface = binary_dilation(1 - bright_surface, iterations=2)
    bright_surface = binary_dilation(1 - bright_surface, iterations=1)

    # Distance-based blending
    blurred = distance(1 - bright_surface).astype(np.float32)
    blurred[blurred > 3] = 3
    blurred = blurred / 3

    # Crop 7-pixel border to match prediction output size
    return blurred[7:-7, 7:-7]


def apply_nodata_mask(
    predictions: np.ndarray,
    s2_stack: np.ndarray,
) -> np.ndarray:
    """Set predictions to 255 (nodata) where input data is missing.

    Args:
        predictions: ``(H, W)`` uint8 tree cover array.
        s2_stack: ``(T, H, W, B)`` Sentinel-2 array used for the prediction.

    Returns:
        ``(H, W)`` uint8 with 255 for nodata pixels.
    """
    # If all timesteps are zero for a pixel, mark as nodata
    all_zero = np.all(s2_stack[..., :4] == 0, axis=(0, -1))

    # Also mark NaN pixels
    any_nan = np.any(np.isnan(s2_stack[..., :4]), axis=(0, -1))

    nodata = all_zero | any_nan
    # Dilate to cover edges
    nodata = binary_dilation(nodata, iterations=10)

    out = predictions.copy()
    # Crop nodata to prediction size if needed
    ph, pw = out.shape
    nodata = nodata[:ph, :pw]
    out[nodata] = 255

    return out
