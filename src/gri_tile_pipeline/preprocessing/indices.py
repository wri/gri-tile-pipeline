"""Vegetation index calculation: EVI, BI, MSAVI2, GRNDVI.

Ported from reference ``preprocessing/indices.py``.
All inputs are ``(T, H, W, B)`` float32 arrays with 10 Sentinel-2 bands.
Band order: Blue, Green, Red, NIR, RE1, RE2, RE3, NIR08, SWIR16, SWIR22.
"""

from __future__ import annotations

import numpy as np


def evi(x: np.ndarray) -> np.ndarray:
    """Enhanced Vegetation Index.

    ``2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)``
    """
    blue = np.clip(x[..., 0], 0, 1)
    red = np.clip(x[..., 2], 0, 1)
    nir = np.clip(x[..., 3], 0, 1)
    val = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    return np.clip(val, -1.5, 1.5)


def bi(x: np.ndarray) -> np.ndarray:
    """Brightness Index.

    ``((SWIR16 + RED) - (NIR + BLUE)) / ((SWIR16 + RED) + (NIR + BLUE))``
    """
    blue = np.clip(x[..., 0], 0, 1)
    red = np.clip(x[..., 2], 0, 1)
    nir = np.clip(x[..., 3], 0, 1)
    swir16 = np.clip(x[..., 8], 0, 1)
    val = ((swir16 + red) - (nir + blue)) / ((swir16 + red) + (nir + blue) + 1e-5)
    return np.clip(val, -1, 1)


def msavi2(x: np.ndarray) -> np.ndarray:
    """Modified Soil-Adjusted Vegetation Index 2.

    ``(2*NIR + 1 - sqrt((2*NIR+1)^2 - 8*(NIR-RED))) / 2``
    """
    red = np.clip(x[..., 2], 0, 1)
    nir = np.clip(x[..., 3], 0, 1)
    sqrt_arg = (2 * nir + 1) ** 2 - 8 * (nir - red)
    sqrt_arg[sqrt_arg < 0] = 0.0
    val = (2 * nir + 1 - np.sqrt(sqrt_arg)) / 2
    return np.clip(val, -1, 1)


def grndvi(x: np.ndarray) -> np.ndarray:
    """Green-Red Normalized Difference Vegetation Index.

    ``(NIR - (GREEN + RED)) / (NIR + (GREEN + RED))``
    """
    green = np.clip(x[..., 1], 0, 1)
    red = np.clip(x[..., 2], 0, 1)
    nir = np.clip(x[..., 3], 0, 1)
    denom = nir + green + red + 1e-5
    return (nir - (green + red)) / denom


def make_indices(arr: np.ndarray) -> np.ndarray:
    """Compute all four indices for ``(T, H, W, B)`` input.

    Returns ``(T, H, W, 4)`` array with [EVI, BI, MSAVI2, GRNDVI].
    """
    out = np.zeros((*arr.shape[:3], 4), dtype=np.float32)
    out[..., 0] = evi(arr)
    out[..., 1] = bi(arr)
    out[..., 2] = msavi2(arr)
    out[..., 3] = grndvi(arr)
    return out
