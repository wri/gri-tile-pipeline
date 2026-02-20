"""Cloud detection, shadow masking, and temporal interpolation.

Ported from reference ``preprocessing/cloud_removal.py`` and
``download_and_predict_job.py``.

Functions fall into three groups:

1. **Simple helpers** kept from the original 72-line file:
   ``hollstein_cloud_mask``, ``id_missing_px``, ``interpolate_na_vals``.

2. **Multi-temporal cloud/shadow detection** (new):
   ``snow_filter``, ``detect_pfcp``, ``identify_clouds_shadows``.

3. **Cloud-free compositing and blending** (new):
   ``id_areas_to_interp``, ``make_aligned_mosaic``,
   ``align_interp_array_lr``, ``calculate_clouds_in_mosaic``,
   ``remove_cloud_and_shadows``.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy import ndimage, signal
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt as distance,
    generate_binary_structure,
    grey_closing,
)

try:
    import bottleneck as bn
except ImportError:  # pragma: no cover
    import numpy as bn  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _water_ndwi(arr: np.ndarray) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR)."""
    return (arr[..., 1] - arr[..., 3]) / (arr[..., 1] + arr[..., 3] + 1e-10)


def _ndbi(arr: np.ndarray) -> np.ndarray:
    """NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)."""
    return (arr[..., 8] - arr[..., 3]) / (arr[..., 8] + arr[..., 3] + 1e-10)


def _ndvi(arr: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red)."""
    return (arr[..., 3] - arr[..., 2]) / (arr[..., 3] + arr[..., 2] + 1e-10)


def _evi(x: np.ndarray) -> np.ndarray:
    """Enhanced vegetation index (EVI)."""
    blue, red, nir = x[..., 0], x[..., 2], x[..., 3]
    return np.clip(2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1), -1.5, 1.5)


def _winsum(in_arr: np.ndarray, windowsize: int) -> np.ndarray:
    """Sum pixels in a moving window (fast cumsum approach)."""
    in_arr = np.pad(in_arr.astype(np.float64), windowsize // 2, mode="reflect")
    in_arr[windowsize:] -= in_arr[:-windowsize]
    in_arr[:, windowsize:] -= in_arr[:, :-windowsize]
    return in_arr.cumsum(0)[windowsize - 1:].cumsum(1)[:, windowsize - 1:]


# ---------------------------------------------------------------------------
# Simple helpers (kept from original file)
# ---------------------------------------------------------------------------

def hollstein_cloud_mask(arr: np.ndarray) -> np.ndarray:
    """Simple cloud detection from Hollstein et al. 2016 (Figure 6).

    Args:
        arr: ``(T, H, W, B)`` Sentinel-2 reflectance array (10 bands,
             float32 in [0, 1]).

    Returns:
        ``(T, H, W)`` boolean cloud mask (True = cloud-free).
    """
    step1 = arr[..., 7] > 0.166
    step2 = arr[..., 1] > 0.21
    step3 = arr[..., 5] / (arr[..., 8] + 1e-10) < 4.292
    cloud = step1 * step2 * step3
    for i in range(cloud.shape[0]):
        cloud[i] = binary_dilation(
            1 - (binary_dilation(cloud[i] == 0, iterations=2)),
            iterations=10,
        )
    return cloud.astype(bool)


def id_missing_px(sentinel2: np.ndarray, thresh: int = 11) -> np.ndarray:
    """Identify time-steps with excessive missing pixels.

    Returns array of integer indices to remove.
    """
    missing_0 = np.sum(sentinel2[..., :10] == 0.0, axis=-1)
    missing_p = np.sum(sentinel2[..., :10] >= 1.0, axis=-1)
    missing = missing_0 + missing_p
    missing = np.sum(missing > 1.0, axis=(1, 2))
    return np.argwhere(missing >= (sentinel2.shape[1] ** 2) / thresh).flatten()


def interpolate_na_vals(s2: np.ndarray) -> np.ndarray:
    """Replace NaN values with temporal median, then 0."""
    if np.sum(np.isnan(s2)) > 0:
        nanmedian = bn.median(s2, axis=0).astype(np.float32)
        nanmedian[np.isnan(nanmedian)] = 0.0
        for t in range(s2.shape[0]):
            nanvals = np.isnan(s2[t])
            s2[t, nanvals] = nanmedian[nanvals]
    return s2


# ---------------------------------------------------------------------------
# Snow filter
# ---------------------------------------------------------------------------

def snow_filter(arr: np.ndarray) -> np.ndarray:
    """Compute snow probability per pixel.

    Reference: ``identify_clouds_shadows`` lines 1554-1576.

    Args:
        arr: ``(T, H, W, 10)`` or ``(H, W, 10)`` float32 S2 data.

    Returns:
        Boolean array of same spatial shape: True = snow.
    """
    ndsi = (arr[..., 1] - arr[..., 8]) / (arr[..., 1] + arr[..., 8] + 1e-10)
    ndsi[ndsi < 0.10] = 0.0
    ndsi[ndsi > 0.42] = 0.42
    snow_prob = (ndsi - 0.1) / 0.32

    # NIR threshold
    snow_prob[arr[..., 3] < 0.10] = 0.0
    snow_prob[np.logical_and(arr[..., 3] > 0.35, snow_prob > 0)] = 1.0

    # Blue threshold
    snow_prob[arr[..., 0] < 0.10] = 0.0
    snow_prob[np.logical_and(arr[..., 0] > 0.22, snow_prob > 0)] = 1.0

    # B2/B4 ratio
    b2b4ratio = arr[..., 0] / (arr[..., 2] + 1e-10)
    snow_prob[b2b4ratio < 0.75] = 0.0

    return snow_prob > 0


# ---------------------------------------------------------------------------
# False positive cloud detection (Fmask 4.0 paralax)
# ---------------------------------------------------------------------------

def detect_pfcp(
    arr: np.ndarray,
    dem: np.ndarray,
    urban_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect potential false-positive cloud pixels (PFCP).

    Uses B8/B8A/B7 paralax (Fmask 4.0) + NDBI/NDVI/NDWI.

    Reference: ``cloud_removal.py`` lines 1109-1212.

    Args:
        arr: ``(T, H, W, 10)`` S2.
        dem: ``(H, W)`` raw DEM in meters (NOT normalized).
        urban_mask: Optional ``(H, W)`` binary mask (1 = urban).
            If *None*, the urban-masking step is skipped.

    Returns:
        ``(fcps, pfps)`` — both ``(T, H, W)`` float arrays.
    """
    from skimage.transform import resize

    ndvi_arr = _ndvi(arr)
    ndwi_arr = _water_ndwi(arr)
    ndbi_arr = _ndbi(arr)
    ndwi_med = np.median(ndwi_arr, axis=0)

    # Per-pixel false positive mask from spectral indices
    pfps = np.logical_and(ndbi_arr > 0, ndbi_arr > ndvi_arr)
    pfps = np.median(pfps.astype(np.float32), axis=0)
    pfps = pfps * (ndwi_med < 0)

    if urban_mask is not None:
        pfps[urban_mask == 1] = 1.0
    else:
        pfps = np.zeros_like(dem, dtype=np.float32)

    pfps[(dem / 90) > 0.10] = 0.0
    pfps = pfps[np.newaxis]
    pfps = np.tile(pfps, (arr.shape[0], 1, 1))

    # B8/B8A/B7 paralax CDI
    cdis = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2]), dtype=np.float32)
    for time in range(arr.shape[0]):
        # Downsample B8
        b8down = np.copy(arr[time, ..., 3])
        if (b8down.shape[0] % 2 + b8down.shape[1] % 2) > 0:
            b8down = resize(
                b8down,
                (b8down.shape[0] + b8down.shape[0] % 2, b8down.shape[1] + b8down.shape[1] % 2),
                order=0, preserve_range=True,
            )
        b8down = ndimage.gaussian_filter(b8down, sigma=0.5, truncate=3)
        b8down = np.reshape(b8down, (b8down.shape[0] // 2, 2, b8down.shape[1] // 2, 2))
        b8down = np.mean(b8down, axis=(1, 3))

        # Downsample B8A
        b8adown = np.copy(arr[time, ..., 7])
        if (b8adown.shape[0] % 2 + b8adown.shape[1] % 2) > 0:
            b8adown = resize(
                b8adown,
                (b8adown.shape[0] + b8adown.shape[0] % 2, b8adown.shape[1] + b8adown.shape[1] % 2),
                order=0, preserve_range=True,
            )
        b8adown = np.reshape(b8adown, (b8adown.shape[0] // 2, 2, b8adown.shape[1] // 2, 2))
        b8adown = np.mean(b8adown, axis=(1, 3))

        # Downsample B7
        b7down = np.copy(arr[time, ..., 6])
        if (b7down.shape[0] % 2 + b7down.shape[1] % 2) > 0:
            b7down = resize(
                b7down,
                (b7down.shape[0] + b7down.shape[0] % 2, b7down.shape[1] + b7down.shape[1] % 2),
                order=0, preserve_range=True,
            )
        b7down = np.reshape(b7down, (b7down.shape[0] // 2, 2, b7down.shape[1] // 2, 2))
        b7down = np.mean(b7down, axis=(1, 3))

        r8a = b8down / (b8adown + 1e-10)
        r8a7 = b7down / (b8adown + 1e-10)

        mean_op = np.ones((7, 7)) / 49.0
        mean_of_sq = signal.convolve2d(r8a ** 2, mean_op, mode="same", boundary="symm")
        sq_of_mean = signal.convolve2d(r8a, mean_op, mode="same", boundary="symm") ** 2
        r8a = mean_of_sq - sq_of_mean

        mean_of_sq = signal.convolve2d(r8a7 ** 2, mean_op, mode="same", boundary="symm")
        sq_of_mean = signal.convolve2d(r8a7, mean_op, mode="same", boundary="symm") ** 2
        r8a7 = mean_of_sq - sq_of_mean

        cdi = (r8a7 - r8a) / (r8a7 + r8a + 1e-10)
        pfcps = (cdi >= -0.4).astype(np.float32)
        pfcps = pfcps.repeat(2, axis=0).repeat(2, axis=1)
        pfcps = resize(pfcps, (arr.shape[1], arr.shape[2]), order=0, preserve_range=True)
        pfcps = pfcps * (_ndvi(arr[time]) < 0.4)
        cdis[time] = pfcps

    struct2 = generate_binary_structure(2, 2)
    for i in range(cdis.shape[0]):
        cdis[i] = binary_dilation(cdis[i], iterations=6, structure=struct2)
        pfps[i] = binary_dilation(pfps[i], iterations=6, structure=struct2)

    fcps = pfps * cdis
    logger.debug(f"PFCP mean per image: {np.mean(fcps, axis=(1, 2))}")
    return fcps, pfps


# ---------------------------------------------------------------------------
# Multi-temporal cloud / shadow detection
# ---------------------------------------------------------------------------

def identify_clouds_shadows(
    img: np.ndarray,
    dem: np.ndarray,
    forest_mask: Optional[np.ndarray] = None,
    urban_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-temporal cloud and shadow detection.

    Reference: ``cloud_removal.py`` lines 1215-1677.

    Args:
        img: ``(T, H, W, 10)`` S2 float32 [0, 1].
        dem: ``(H, W)`` raw DEM in meters (NOT ``/90``).
        forest_mask: Optional ``(H, W)`` binary. Defaults to zeros.
        urban_mask: Optional ``(H, W)`` binary. Passed to ``detect_pfcp``.

    Returns:
        ``(clouds, fcps)`` — ``(T, H, W)`` float arrays.
    """
    logger.info("Computing multi-temporal cloud/shadow mask")

    if forest_mask is None:
        forest_mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.float32)

    water_mask = bn.nanmedian(_water_ndwi(img), axis=0)
    shadows = np.zeros_like(img[..., 0], dtype=np.float32)
    clouds = np.zeros_like(shadows, dtype=np.float32)

    # Initial Hollstein cloud mask for shadow detection
    def _hollstein_cld(arr):
        step1 = arr[..., 7] > 0.166
        step2b = arr[..., 1] > 0.28
        step3 = arr[..., 5] / (arr[..., 8] + 1e-10) < 4.292
        cl = step1 * step2b * step3
        for i in range(cl.shape[0]):
            cl[i] = binary_dilation(1 - (binary_dilation(cl[i] == 0, iterations=2)), iterations=10)
        return cl

    clm = _hollstein_cld(img)

    # --- Shadow detection (per timestep) ---
    for time in range(img.shape[0]):
        lower = max(0, time - 4)
        upper = min(img.shape[0], time + 3)
        if (upper - lower) == 3:
            if upper == img.shape[0]:
                lower = max(lower - 1, 0)
            if lower == 0:
                upper = min(upper + 1, img.shape[0])
        others = np.arange(lower, upper)
        ri_shadow = np.copy(img[..., [0, 1, 7, 8]])
        ri_shadow = ri_shadow[others]
        ri_shadow[clm[others] > 0] = np.nan
        ri_shadowmax = bn.nanmax(ri_shadow, axis=0)
        ri_shadow_med = bn.nanmedian(ri_shadow, axis=0)
        ri_shadow_med[np.isnan(ri_shadow_med)] = np.min(
            img[..., [0, 1, 7, 8]], axis=0
        )[np.isnan(ri_shadow_med)]

        deltab8a = (img[time, ..., 7] - ri_shadow_med[..., 2]) < -0.04
        deltab11 = (img[time, ..., 8] - ri_shadow_med[..., 3]) < -0.04
        ti0 = img[time, ..., 0] < 0.09
        deltablue = (img[time, ..., 0] - ri_shadow_med[..., 0]) < -0.02
        shadows_i = deltab11 * deltab8a * ti0 * deltablue * (img[time, ..., 7] < 0.17)

        deltab8a = (img[time, ..., 7] - ri_shadowmax[..., 2]) < -0.04
        deltab11 = (img[time, ..., 8] - ri_shadowmax[..., 3]) < -0.04
        shadows_i_dark = deltab11 * deltab8a * (img[time, ..., 0] < 0.03) * (img[time, ..., 7] < 0.18)
        shadows_i_dark[water_mask > 0] = 0.0
        shadows_i = np.maximum(shadows_i, shadows_i_dark)
        shadows_i[water_mask > 0] = 0.0

        # Slope-relaxed shadows
        ri_shadow_full = np.copy(img[..., [0, 1, 7, 8]])
        ri_shadow_full[clm > 0] = np.nan
        ri_shadow_full = bn.nanmedian(ri_shadow_full, axis=0)
        ri_shadow_full[np.isnan(ri_shadow_full)] = np.median(
            img[..., [0, 1, 7, 8]], axis=0
        )[np.isnan(ri_shadow_full)]

        deltab8a = (img[time, ..., 7] - ri_shadowmax[..., 2]) < -0.04
        deltab11 = (img[time, ..., 8] - ri_shadowmax[..., 3]) < -0.04
        shadows_slope = deltab8a * deltab11 * (img[time, ..., 0] < 0.07) * (img[time, ..., 7] < 0.18)
        shadows_slope = shadows_slope * (np.sum(img[time, ..., :3], axis=-1) < 0.28)
        shadows_slope[water_mask > 0] = 0.0
        shadows_slope = shadows_slope * (dem >= 25)
        shadows_i = np.maximum(shadows_i, shadows_slope)

        # Water shadows
        water_shadow = (
            ((img[time, ..., 0] - ri_shadow_full[..., 0]) < -0.05)
            * ((img[time, ..., 1] - ri_shadow_full[..., 1]) < -0.05)
            * (img[time, ..., 7] < 0.03)
            * ((ri_shadow_full[..., 1] - img[time, ..., 1]) > 0.02)
            * (water_mask > 0)
        )
        shadows[time] = shadows_i + water_shadow

    # Erode/dilate shadow mask
    for i in range(shadows.shape[0]):
        shadows_i = binary_dilation(
            1 - (binary_dilation(shadows[i] == 0, iterations=2)), iterations=3
        )
        shadows_i = distance(1 - shadows_i)
        shadows_i[shadows_i <= 5] = 0.0
        shadows_i[shadows_i > 5] = 1.0
        shadows[i] = 1 - shadows_i

    # --- Cloud detection (per timestep) ---
    for time in range(img.shape[0]):
        lower = max(0, time - 2)
        upper = min(img.shape[0], time + 3)
        if (upper - lower) == 3:
            if upper == img.shape[0]:
                lower = max(lower - 2, 0)
            if lower == 0:
                upper = min(upper + 2, img.shape[0])
        others = np.arange(lower, upper)
        close = [max(0, time - 1), min(img.shape[0] - 1, time + 1)]
        if close[1] - close[0] < 2:
            if close[0] == 0:
                close[0] += 1
                close[1] += 1
            else:
                close[1] -= 1
                close[0] -= 1
        if len(close) == 2:
            if close[-1] >= (img.shape[0] - 2) and img.shape[0] > 3:
                close = np.concatenate([np.array([close[0] - 1]), close])

        ri_ref = np.copy(img[..., [0, 1, 2]])
        if img.shape[0] > 2:
            ri_ref[shadows > 0] = np.nan
            ri_upper0 = bn.nanmin(ri_ref[others, ..., 0], axis=0)
            ri_upper1 = bn.nanmin(ri_ref[others, ..., 1], axis=0)
            ri_upper2 = bn.nanmin(ri_ref[others, ..., 2], axis=0)
            nan_replace = np.isnan(ri_upper0)
            ri_upper0[nan_replace] = np.percentile(img[..., 0], 25, axis=0)[nan_replace]
            ri_upper1[nan_replace] = np.percentile(img[..., 1], 25, axis=0)[nan_replace]
            ri_upper2[nan_replace] = np.percentile(img[..., 2], 25, axis=0)[nan_replace]
            ri_close = bn.nanmin(ri_ref[close], axis=0).astype(np.float32)

            min_i = close[0]
            max_i = close[-1]
            for _iteration in range(10):
                if np.sum(np.isnan(ri_close)) > 0:
                    min_i = max(min_i - 1, 0)
                    max_i = min(max_i + 1, img.shape[0])
                    close_exp = np.array([x for x in range(min_i, max_i) if x != time])
                    ri_close_new = bn.nanmin(ri_ref[close_exp], axis=0).astype(np.float32)
                    ri_close[np.isnan(ri_close)] = ri_close_new[np.isnan(ri_close)]

            if np.sum(np.isnan(ri_close)) > 0:
                ri_close[np.isnan(ri_close)] = np.min(img[..., :3], axis=0)[np.isnan(ri_close)]
        else:
            ri_close = np.min(ri_ref, axis=0).astype(np.float32)
            ri_upper0 = ri_close[..., 0]
            ri_upper1 = ri_close[..., 1]
            ri_upper2 = ri_close[..., 2]

        close_thresh = np.minimum(((ri_close[..., 0] / 0.02 / 100) + 0.005), 0.10)
        close_thresh = np.maximum(close_thresh, 0.05)
        close_thresh[forest_mask == 1] -= 0.02
        close_thresh = np.maximum(close_thresh, 0.04)

        clouds_mean = 0.0
        clouds_close_mean = 1.0
        close_modifier = 0.0
        while (clouds_close_mean - clouds_mean) > 0.075:
            deltab2 = (img[time, ..., 0] - ri_upper0) > 0.08
            deltab3 = (img[time, ..., 1] - ri_upper1) > 0.08
            deltab4 = (img[time, ..., 2] - ri_upper2) > 0.07
            closeb2 = (img[time, ..., 0] - ri_close[..., 0]) > (close_thresh + close_modifier + 0.01)
            closeb3 = (img[time, ..., 1] - ri_close[..., 1]) > (close_thresh + close_modifier + 0.01)
            closeb4 = (img[time, ..., 2] - ri_close[..., 2]) > (close_thresh + close_modifier)
            clouds_i = deltab2 * deltab3 * deltab4
            clouds_close = closeb2 * closeb3 * closeb4
            clouds_mean = np.mean(clouds_i > 0)
            clouds_close_mean = np.mean(clouds_close > 0)
            close_modifier += 0.0025

        brightness = np.sum(img[time, ..., :3], axis=-1) < 0.75
        clouds_close = clouds_close * brightness
        clouds_close_nonforest = 1 - (binary_dilation(clouds_close == 0, iterations=2))
        clouds_close[forest_mask == 0] = clouds_close_nonforest[forest_mask == 0]
        clouds[time] = np.maximum(clouds_i, clouds_close)

    # --- Multitemporal brightness z-score clouds ---
    brightness_mask = np.sum(img[..., :3], axis=-1)
    brightness_mask[np.logical_or(clouds > 0, shadows > 0)] = np.nan
    median_brightness = np.nanmedian(brightness_mask, axis=(1, 2))

    brightness_clouds = np.zeros_like(clouds, dtype=np.float32)
    for i in range(img.shape[0]):
        brightness_i = np.sum(img[i, ..., :3], axis=-1)
        brightness_ratio = brightness_i / (median_brightness[i] + 1e-10)
        brightness_ratio[water_mask > 0] = 1.0
        if np.sum(clouds[i]) < 0.90 * clouds[i].size:
            brightness_zscore = (
                brightness_ratio - np.nanmean(brightness_ratio[clouds[i] == 0])
            ) / (np.nanstd(brightness_ratio[clouds[i] == 0]) + 1e-10)
        else:
            brightness_zscore = (
                brightness_ratio - np.nanmean(brightness_ratio)
            ) / (np.nanstd(brightness_ratio) + 1e-10)
        brightness_clouds[i][brightness_zscore > 3.5] = 1.0
        brightness_clouds[i] *= (water_mask < 0)

    sum_brightness_clouds = np.sum((brightness_clouds - clouds) > 0, axis=0)
    for i in range(img.shape[0]):
        brightness_clouds[i][sum_brightness_clouds > 1] = 0.0
    clouds = np.maximum(clouds, brightness_clouds)

    # Remove non-white bright surfaces
    for i in range(clouds.shape[0]):
        mean_brightness = np.mean(img[i, ..., :3], axis=-1)
        is_possible_fp = mean_brightness < 0.4
        vis_range = np.max(img[i, ..., :3], axis=-1) - np.min(img[i, ..., :3], axis=-1)
        is_fp = is_possible_fp * ((vis_range / (mean_brightness + 1e-10)) > 0.5)
        clouds[i] = clouds[i] * (1 - is_fp)

    # --- False positive cloud removal (Fmask 4.0 paralax) ---
    fcps, pfcps = detect_pfcp(img, dem, urban_mask=urban_mask)

    for i in range(clouds.shape[0]):
        mini = max(i - 1, 0)
        maxi = min(i + 2, img.shape[0])
        brightness = np.min(img[mini:maxi, ..., :3], axis=(0, 3))
        brightnessi = np.mean(img[i, ..., :3], axis=-1)
        isnt_cloud = (brightnessi - brightness) < 0.4
        toremove = np.logical_and(fcps[i] > 0, isnt_cloud)
        clouds[i][toremove] = 0.0
        shadows[i][toremove] = 0.0

    # NIR/SWIR ratio false positive removal
    nir_swir_ratio = img[..., 3] / (img[..., 8] + 0.01)
    nir_swir_ratio = nir_swir_ratio < 0.75
    nir_swir_ratio = binary_dilation(nir_swir_ratio, iterations=3)
    for i in range(clouds.shape[0]):
        mini = max(i - 1, 0)
        maxi = min(i + 2, img.shape[0])
        brightness = np.min(img[mini:maxi, ..., :3], axis=(0, 3))
        brightnessi = np.mean(img[i, ..., :3], axis=-1)
        isnt_cloud = ((brightnessi - brightness) < 0.4)
        nir_swir_ratio[i][water_mask < 0] = 0.0
        clouds[i][np.logical_and(nir_swir_ratio[i] > 0, isnt_cloud)] = 0.0

    # Water false positive removal
    for i in range(img.shape[0]):
        fp = (water_mask > 0) * (img[i, ..., 8] < 0.11)
        fp = binary_dilation(fp, iterations=10)
        clouds[i][fp] = 0.0

    # Window majority filter
    for i in range(clouds.shape[0]):
        window_sum = _winsum(clouds[i], 3)
        clouds[i][window_sum < 5] = 0.0

    # Remove very dark false positives
    for i in range(clouds.shape[0]):
        brightness_threshold = np.sum(img[i, ..., :3], axis=-1) < 0.21
        brightness_threshold = binary_dilation(brightness_threshold, iterations=3)
        brightness_threshold = brightness_threshold * (1 - forest_mask)
        clouds[i][brightness_threshold.astype(bool)] = 0.0

    # Dilate clouds
    struct2 = generate_binary_structure(2, 2)
    for i in range(clouds.shape[0]):
        clouds[i] = 1 - (binary_dilation(clouds[i] == 0, iterations=1))
        pfcps[i] = binary_dilation(pfcps[i], iterations=5)
        urban_clouds = clouds[i] * pfcps[i]
        urban_clouds = 1 - (binary_dilation(urban_clouds == 0, iterations=3))

        non_urban_clouds = clouds[i] * (1 - pfcps[i])
        window_sum = _winsum(non_urban_clouds, 3)
        is_large_cloud = np.copy(non_urban_clouds)
        is_large_cloud[window_sum < 6] = 0.0
        is_small_cloud = np.copy(non_urban_clouds)
        is_small_cloud[window_sum >= 6] = 0.0
        is_small_cloud = binary_dilation(is_small_cloud, iterations=1)
        is_large_cloud = binary_dilation(is_large_cloud, iterations=5)
        non_urban_clouds = np.maximum(is_large_cloud, is_small_cloud)

        non_urban_clouds = distance(1 - non_urban_clouds)
        non_urban_clouds[non_urban_clouds <= 3] = 0.0
        non_urban_clouds[non_urban_clouds > 3] = 1.0
        non_urban_clouds = 1 - non_urban_clouds
        clouds[i] = non_urban_clouds + urban_clouds

    # Shadow proximity filtering
    for i in range(clouds.shape[0]):
        if np.mean(shadows[i]) > (np.mean(clouds[i]) + 0.3):
            if np.mean(clouds[i]) < 0.3:
                dilated = binary_dilation(np.copy(clouds[i]), iterations=50)
                dilated = np.logical_or(dilated, dem >= 30)
                shadows[i] = shadows[i] * dilated
        if np.mean(clouds[i]) < 0.05 and np.mean(clouds[i]) > 0:
            if (np.mean(shadows[i]) / (np.mean(clouds[i]) + 1e-10)) > 3:
                dilated = binary_dilation(np.copy(clouds[i]), iterations=50)
                dilated = np.logical_or(dilated, dem >= 30)
                shadows[i] = shadows[i] * dilated

    clouds = np.maximum(clouds, shadows)
    fcps = np.maximum(fcps, nir_swir_ratio)
    fcps = binary_dilation(fcps, iterations=2)

    # Inverse-blue false negative shadows
    for i in range(clouds.shape[0]):
        if np.mean(clouds[i]) < 0.9:
            blue_i = np.copy(img[i, ..., 0])
            blue_i = blue_i[clouds[i] == 0]
            if len(blue_i) > 0:
                reference = np.mean(1 / (blue_i + 1e-10)) + 2 * np.std(1 / (blue_i + 1e-10))
                shadow_i = 1 / (img[i, ..., 0] + 1e-10) > reference
                shadow_i = shadow_i * (img[i, ..., 7] < 0.17)
                shadow_i = binary_dilation(
                    1 - (binary_dilation(shadow_i == 0, iterations=2)), iterations=2
                )
                shadow_i[water_mask > 0] = 0.0
                clouds[i] = np.maximum(clouds[i], shadow_i)
    clouds[clouds > 1] = 1.0

    # --- Haze detection ---
    mean_brightness = np.mean(img[..., :3], axis=-1)
    mean_cloudfree_brightness = []
    std_cloudfree_brightness = []
    std_cloudfree_whiteness = []
    for i in range(clouds.shape[0]):
        if np.mean(clouds[i]) < 1:
            imi = img[i, ..., :3]
            imi = imi[clouds[i] == 0]
            mean_cloudfree_brightness.append(np.mean(mean_brightness[i][clouds[i] == 0]))
            std_cloudfree_brightness.append(np.std(mean_brightness[i][clouds[i] == 0]))
            std_cloudfree_whiteness.append(np.std(np.ptp(imi, axis=1)))

    if len(mean_cloudfree_brightness) > 0:
        median_bright = np.median(mean_cloudfree_brightness)
        median_std = np.median(std_cloudfree_brightness)
        haze_brightness = np.array(mean_cloudfree_brightness) / (median_bright + 1e-10)
        haze_std = np.array(std_cloudfree_brightness) / (median_std + 1e-10)
        haze_whiteness = np.array(std_cloudfree_whiteness) / (np.median(std_cloudfree_whiteness) + 1e-10)
        haze = (haze_brightness >= 1.5) * (haze_std <= 0.67) * (haze_whiteness < 1)
        haze = np.logical_or(haze, (haze_brightness >= 1.3) * (haze_std <= 0.5))
        logger.debug(f"Haze flags: {haze}")
        j = 0
        for i in range(clouds.shape[0]):
            if np.mean(clouds[i]) < 1:
                if haze[j]:
                    clouds[i] = 1.0
                j += 1

    cloud_pct = np.mean(clouds, axis=(1, 2)) * 100
    logger.info(f"Cloud+shadow % per image: {np.around(cloud_pct, 1)}")
    return clouds, fcps


# ---------------------------------------------------------------------------
# Interpolation mask
# ---------------------------------------------------------------------------

def id_areas_to_interp(
    probs: np.ndarray,
    **_kwargs,
) -> np.ndarray:
    """Build soft interpolation masks from cloud probability maps.

    Reference: ``cloud_removal.py`` lines 774-798.

    Args:
        probs: ``(T, H, W)`` cloud/shadow probability (0..1).

    Returns:
        ``(T, H, W)`` float32 interpolation weights in [0, 1].
    """
    areas = np.copy(probs).astype(np.float32)
    areas = np.clip(areas, 0, 1)

    for date in range(areas.shape[0]):
        if np.sum(areas[date]) > 0:
            blurred = distance(1 - areas[date])
            blurred[blurred > 12] = 12
            blurred = blurred / 12.0
            blurred = 1 - blurred
            blurred[blurred < 0.2] = 0.0
            blurred = grey_closing(blurred, size=15)
            areas[date] = blurred

    return areas.astype(np.float32)


# ---------------------------------------------------------------------------
# Cloud-free mosaic (linear normalization path)
# ---------------------------------------------------------------------------

def make_aligned_mosaic(arr: np.ndarray, interp: np.ndarray) -> np.ndarray:
    """Build a radiometrically normalised cloud-free mosaic.

    Uses linear mean/std normalization (``randomforest=False`` branch).

    Reference: ``cloud_removal.py`` lines 578-699.

    Args:
        arr: ``(T, H, W, 10)`` S2.
        interp: ``(T, H, W)`` interpolation weights.

    Returns:
        ``(H, W, 10)`` cloud-free mosaic.
    """
    water_mask = np.median(
        (arr[..., 1] - arr[..., 3]) / (arr[..., 1] + arr[..., 3] + 1e-10), axis=0
    )
    water_mask = water_mask > 0
    water_mask = binary_dilation(1 - water_mask, iterations=2)
    water_mask = binary_dilation(1 - water_mask, iterations=5)

    mosaic = np.zeros((arr.shape[1], arr.shape[2], arr.shape[3]), dtype=np.float32)
    divisor = np.sum(1 - interp, axis=0)[..., np.newaxis]

    for i in range(arr.shape[0]):
        non_interp_mosaic_mask = np.logical_and(interp[i] < 0.25, water_mask == 0)
        non_interp_areas = np.full_like(arr[0], 0, dtype=np.float32)
        non_interp_count = np.full_like(arr[0], 0, dtype=np.float32)

        for b in range(arr.shape[0]):
            if b != i:
                mask = np.logical_and(
                    np.logical_and(interp[i] < 0.25, interp[b] < 1), water_mask == 0
                )
                arr_b = arr[b]
                combined = mask * non_interp_mosaic_mask
                non_interp_areas[combined] += arr_b[combined]
                non_interp_count[combined] += 1

        non_interp_areas = non_interp_areas / (non_interp_count + 1e-10)
        non_interp_mosaic_mask[non_interp_count[..., 0] == 0] = False
        non_interp_mosaic = arr[i][non_interp_mosaic_mask]
        non_interp_areas_flat = np.reshape(
            non_interp_areas,
            (non_interp_areas.shape[0] * non_interp_areas.shape[1], non_interp_areas.shape[2]),
        )
        non_interp_areas_flat = non_interp_areas_flat[~np.isnan(non_interp_areas_flat).any(axis=1)]

        if non_interp_mosaic.shape[0] > 1000 and non_interp_areas_flat.shape[0] > 1000:
            n_min = min(non_interp_mosaic.shape[0], non_interp_areas_flat.shape[0])
            non_interp_mosaic = non_interp_mosaic[:n_min]
            non_interp_areas_flat = non_interp_areas_flat[:n_min]

            mean_ref = bn.nanmedian(non_interp_areas_flat, axis=0)
            std_ref = bn.nanstd(non_interp_areas_flat, axis=0)
            mean_src = bn.nanmedian(non_interp_mosaic, axis=0)
            std_src = bn.nanstd(non_interp_mosaic, axis=0)

            std_mult = std_ref / (std_src + 1e-10)
            addition = mean_ref - (mean_src * std_mult)
            arr_i = np.copy(arr[i])
            arr_i[water_mask == 0] = arr_i[water_mask == 0] * std_mult + addition
            increment = (1 - interp[i][..., np.newaxis]) * arr_i
            mosaic = mosaic + increment
        elif np.mean(water_mask) < 0.9:
            interp[i] = 1.0
        else:
            continue

    divisor[divisor < 0] = 0.0
    mosaic = mosaic / (divisor + 1e-10)
    mosaic[np.isnan(mosaic)] = np.percentile(arr, 10, axis=0)[np.isnan(mosaic)]
    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)
    mosaic = np.maximum(mosaic, mins)
    mosaic = np.minimum(mosaic, maxs)
    return mosaic


# ---------------------------------------------------------------------------
# Post-hoc cloud detection in mosaic
# ---------------------------------------------------------------------------

def calculate_clouds_in_mosaic(
    mosaic: np.ndarray,
    interp: np.ndarray,
    pfcps: np.ndarray,
) -> np.ndarray:
    """Detect residual clouds in the mosaic.

    Reference: ``cloud_removal.py`` lines 703-732.
    """
    only_1_img = np.sum(1 - (interp > 0), axis=0).squeeze() < 2
    if len(pfcps.shape) == 3 and pfcps.shape[0] > 1:
        pfcps = pfcps[0]
    pfcps = binary_dilation(pfcps, iterations=10)

    only_1_img = np.maximum(only_1_img, pfcps.squeeze())
    if np.sum(only_1_img) == np.prod(only_1_img.shape):
        return np.zeros_like(only_1_img)

    reference_blue = np.percentile(mosaic[..., 0][~only_1_img], 99)
    reference_red = np.percentile(mosaic[..., 2][~only_1_img], 99)
    clouds_in_mosaic = (
        (mosaic[..., 0] > reference_blue)
        * (mosaic[..., 2] > reference_red)
        * only_1_img
        * (np.sum(mosaic[..., :3], axis=-1) < 1)
    )
    clouds_in_mosaic[pfcps.squeeze() > 0] = 0.0
    clouds_in_mosaic = binary_dilation(1 - clouds_in_mosaic, iterations=3)
    clouds_in_mosaic = binary_dilation(1 - clouds_in_mosaic, iterations=8)
    return clouds_in_mosaic


# ---------------------------------------------------------------------------
# Per-date linear regression alignment
# ---------------------------------------------------------------------------

def align_interp_array_lr(
    interp_array: np.ndarray,
    array: np.ndarray,
    date: int,
    interp: np.ndarray,
    mosaic: np.ndarray,
    water_mask: np.ndarray,
) -> Tuple[np.ndarray, list]:
    """Normalise interpolated areas to non-interpolated areas via OLS.

    Reference: ``align_interp_array_randomforest`` lines 316-575.
    Uses ``LinearRegression`` + equibatch sampling (active code path).

    Args:
        interp_array: ``(1, H, W, 10)`` mosaic values for areas to blend.
        array: ``(T, H, W, 10)`` full S2 array.
        date: Current date index.
        interp: ``(T, H, W)`` interpolation weights.
        mosaic: ``(H, W, 10)`` cloud-free mosaic.
        water_mask: ``(H, W)`` water mask.

    Returns:
        ``(interp_array, to_remove)`` — adjusted array and list of dates to remove.
    """
    from sklearn.linear_model import LinearRegression

    snow = np.mean(snow_filter(array).astype(np.float32), axis=0)[..., np.newaxis]
    to_remove: list = []

    if np.sum(interp[date] > 0) > 0 and np.sum(interp[date] == 0) > 0:
        if np.mean(np.logical_and(interp[date] < 1, water_mask <= 1)) > 0.01:
            array_i = np.copy(array[date])
            interp_array_i = np.copy(interp_array[0])

            n_current_time = np.sum(np.logical_and(interp[date] == 0, water_mask <= 1))

            if n_current_time > 40000:
                min_time = max(date, 0)
                max_time = date + 1
            else:
                if date == (array.shape[0] - 1):
                    min_time = max(date - 2, 0)
                else:
                    min_time = max(date - 1, 0)
                max_time = min(date + 2, array.shape[0])

            non_interp_areas = []
            non_interp_mosaic = []

            n_current_time = max(n_current_time, 36000)
            for t in range(min_time, max_time):
                requirement1 = np.logical_and(interp[t] == 0, water_mask < 1)
                non_interp_areasi = np.concatenate([array[t], snow], axis=-1)[requirement1]
                non_interp_mosaici = np.concatenate([mosaic, snow], axis=-1)[requirement1]
                non_interp_areas.append(non_interp_areasi)
                non_interp_mosaic.append(non_interp_mosaici)

            if n_current_time > 40000 and len(non_interp_mosaic) > 0:
                non_interp_mid_mosaic = non_interp_mosaic[0]
                non_interp_mid_areas = non_interp_areas[0]
            elif len(non_interp_mosaic) > 0:
                non_interp_mid_mosaic = np.concatenate(non_interp_mosaic, axis=0)
                non_interp_mid_areas = np.concatenate(non_interp_areas, axis=0)
            else:
                return interp_array[0], to_remove

            # Equibatch sampling by EVI quintiles
            n_samples = min(90000, non_interp_mid_mosaic.shape[0])
            n_samples_i = n_samples // 5
            ndvi_i = _evi(non_interp_mid_areas)

            b2 = np.percentile(ndvi_i, 2)
            b20 = np.percentile(ndvi_i, 20)
            b40 = np.percentile(ndvi_i, 40)
            b60 = np.percentile(ndvi_i, 60)
            b80 = np.percentile(ndvi_i, 80)
            b98 = np.percentile(ndvi_i, 98)

            p2 = np.argwhere(ndvi_i < b2).squeeze()
            p20 = np.argwhere(ndvi_i < b20).squeeze()
            p40 = np.argwhere(np.logical_and(ndvi_i >= b20, ndvi_i < b40)).squeeze()
            p60 = np.argwhere(np.logical_and(ndvi_i >= b40, ndvi_i < b60)).squeeze()
            p80 = np.argwhere(np.logical_and(ndvi_i >= b60, ndvi_i < b80)).squeeze()
            p100 = np.argwhere(ndvi_i >= b80).squeeze()
            p98 = np.argwhere(ndvi_i >= b98).squeeze()

            # Ensure 1-D arrays for repeat/shuffle
            for arr_ref in [p2, p20, p40, p60, p80, p100, p98]:
                if arr_ref.ndim == 0:
                    arr_ref = arr_ref.reshape(1)

            p98 = np.repeat(p98, 10)
            p2 = np.repeat(p2, 10)
            for a in [p2, p98, p20, p40, p60, p80, p100]:
                random.shuffle(a)

            p20 = p20[:n_samples_i]
            p40 = p40[:n_samples_i]
            p60 = p60[:n_samples_i]
            p80 = p80[:n_samples_i]
            p100 = p100[:n_samples_i]
            random_sample = np.concatenate([p2, p20, p40, p60, p80, p100, p98])
            random.shuffle(random_sample)

            random_sample = random_sample[: non_interp_mid_mosaic.shape[0]]
            random_sample = random_sample[: non_interp_mid_areas.shape[0]]
            non_interp_mid_mosaic = non_interp_mid_mosaic[random_sample]
            non_interp_mid_areas = non_interp_mid_areas[random_sample]

            # Per-band linear regression
            preds_out = np.copy(interp_array_i)
            for band in range(10):
                train_x = np.copy(non_interp_mid_mosaic)
                non_interp_mid_mosaic_clipped = np.copy(non_interp_mid_mosaic)
                non_interp_mid_mosaic_clipped[..., band] = np.clip(
                    non_interp_mid_mosaic_clipped[..., band], 0.005, 1
                )

                predicted = np.copy(np.concatenate([interp_array_i, snow], axis=-1))
                predicted = np.reshape(
                    predicted, (predicted.shape[0] * predicted.shape[1], predicted.shape[-1])
                )

                model = LinearRegression(positive=True, fit_intercept=False).fit(
                    train_x, non_interp_mid_areas[..., band]
                )
                predicted = model.predict(predicted)
                predicted = np.reshape(predicted, interp_array_i.shape[:-1])

                mask = np.logical_and(interp[date] > 0, water_mask <= 1)
                preds_out[mask, band] = predicted[mask]

            interp_array = preds_out
        else:
            interp_array = interp_array[0]
    else:
        to_remove = []
        interp_array = interp_array[0]

    return interp_array, to_remove


# ---------------------------------------------------------------------------
# Orchestrator: full cloud removal
# ---------------------------------------------------------------------------

def remove_cloud_and_shadows(
    tiles: np.ndarray,
    probs: np.ndarray,
    fcps: np.ndarray,
    s1: Optional[np.ndarray] = None,
    mosaic: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Remove clouds and shadows by blending with a cloud-free mosaic.

    Reference: ``cloud_removal.py`` lines 888-973.

    Args:
        tiles: ``(T, H, W, 10)`` S2 data.
        probs: ``(T, H, W)`` cloud/shadow probability.
        fcps: ``(T, H, W)`` false positive cloud pixels from ``detect_pfcp``.
        s1: Unused, kept for interface compatibility.
        mosaic: Optional pre-computed mosaic. If *None*, computed internally.

    Returns:
        ``(tiles, areas_interpolated, to_remove)`` — cleaned tiles, interp
        weights, and list of fully-cloudy date indices.
    """
    areas_interpolated = np.copy(probs).astype(np.float32)

    for date in range(areas_interpolated.shape[0]):
        if np.sum(areas_interpolated[date]) > 0:
            blurred = distance(1 - areas_interpolated[date])
            blurred[blurred > 12] = 12
            blurred = blurred / 12.0
            blurred = 1 - blurred
            blurred[blurred < 0.2] = 0.0
            blurred = grey_closing(blurred, size=20)
            areas_interpolated[date] = blurred

    areas_interpolated = areas_interpolated.astype(np.float32)

    if mosaic is None:
        mosaic = make_aligned_mosaic(tiles, areas_interpolated)

    water_mask = (_water_ndwi(np.median(tiles, axis=0)) > 0.0).astype(np.float32)
    to_remove: List[int] = []

    logger.info("Blending mosaic and cloud-free portions")
    for date in range(tiles.shape[0]):
        interp_array = np.zeros_like(tiles[date])
        interp_array[areas_interpolated[date] > 0] = mosaic[areas_interpolated[date] > 0]

        interp_array, removei = align_interp_array_lr(
            interp_array[np.newaxis],
            tiles,
            date,
            areas_interpolated,
            mosaic,
            water_mask,
        )

        tiles[date] = (
            tiles[date] * (1 - areas_interpolated[date][..., np.newaxis])
            + interp_array * areas_interpolated[date][..., np.newaxis]
        )
        if len(removei) > 0:
            to_remove.append(date)
        if np.mean(areas_interpolated[date] == 1) == 1:
            to_remove.append(date)

    clouds_in_mosaic = calculate_clouds_in_mosaic(
        mosaic, areas_interpolated.squeeze(), fcps
    )
    areas_interpolated += clouds_in_mosaic[np.newaxis]
    areas_interpolated[areas_interpolated > 1] = 1.0

    return tiles, areas_interpolated, to_remove
