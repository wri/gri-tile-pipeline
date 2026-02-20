"""Overlapping window prediction with Gaussian blending.

Ported from reference ``download_and_predict_job.py`` lines 1412-1600
and the ``load_mosaic_predictions`` function (lines 1609-1761).

Reference size conventions:
  SIZE = 172 - 14 = 158   (output crop size, stored in global SIZE)
  graph input  = SIZE + 14 = 172   (border = 7 on each side)
  graph file   = predict_graph-172.pb
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from gri_tile_pipeline.inference.normalize import normalize_subtile


# Output crop size (matches reference global SIZE = 172-14)
SIZE = 158
# Border pixels on each side for overlap
BORDER = 7
# Full model input size
INPUT_SIZE = SIZE + 2 * BORDER  # 172


def fspecial_gauss(size: int, sigma: float) -> np.ndarray:
    """Generate a 2-D Gaussian kernel (mimics MATLAB ``fspecial('gaussian',...)``)."""
    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g


def predict_subtile(
    subtile: np.ndarray,
    predict_session,
    output_size: int = SIZE,
    length: int = 4,
) -> np.ndarray:
    """Run TF inference on a single ``(T, H, W, 17)`` subtile.

    Args:
        subtile: Feature stack, already normalized to [-1, 1].
        predict_session: :class:`PredictSession` from ``frozen_graph.py``.
        output_size: Expected spatial output (158 for the 172-input graph).
        length: Temporal sequence length (passed to PlaceholderWithDefault).

    Returns:
        ``(output_size, output_size)`` float32 predictions in [0, 1].
    """
    if np.sum(subtile) == 0:
        return np.full((output_size, output_size), 255, dtype=np.float32)

    if not np.issubdtype(subtile.dtype, np.floating):
        subtile = subtile / 65535.0

    batch_x = subtile[np.newaxis].astype(np.float32)
    lengths = np.full((batch_x.shape[0],), length)

    preds = predict_session.sess.run(
        predict_session.logits,
        feed_dict={
            predict_session.inp: batch_x,
            predict_session.length: lengths,
        },
    )
    preds = preds.squeeze()

    # Crop to output_size (dynamic — model output may differ from input)
    clip = (preds.shape[0] - output_size) // 2
    if clip > 0:
        preds = preds[clip:-clip, clip:-clip]

    return preds.astype(np.float32)


def make_windows(
    height: int,
    width: int,
    tile_size: int = SIZE,
    n_rows: int = 6,
    margin: int = BORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate overlapping window coordinates for prediction.

    Replicates the reference ``tiles_folder`` / ``tiles_array`` logic:
    - ``tiles_folder``: non-overlapping grid positions (SIZE x SIZE)
    - ``tiles_array``: expanded by ``margin`` on each side for overlap

    Returns:
        ``(tiles_folder, tiles_array)`` — each ``(N, 4)`` arrays of
        ``[x_start, y_start, width, height]``.
    """
    gap_x = int(np.ceil((height - tile_size) / max(n_rows - 1, 1)))
    gap_y = int(np.ceil((width - tile_size) / max(n_rows - 1, 1)))

    xs = np.hstack([np.arange(0, height - tile_size, gap_x), [height - tile_size]])
    ys = np.hstack([np.arange(0, width - tile_size, gap_y), [width - tile_size]])

    # Cartesian product
    mesh = np.meshgrid(xs, ys)
    windows = np.column_stack([m.ravel() for m in mesh])
    sizes = np.full_like(windows, tile_size)
    tiles_folder = np.hstack([windows, sizes])

    # Expand by margin for overlap (tiles_array)
    tiles_array = np.copy(tiles_folder)
    tiles_array[:, 0] = np.maximum(0, tiles_array[:, 0] - margin)
    tiles_array[:, 1] = np.maximum(0, tiles_array[:, 1] - margin)
    tiles_array[:, 2] = tile_size + 2 * margin
    tiles_array[:, 3] = tile_size + 2 * margin
    # Clip to spatial bounds
    tiles_array[:, 2] = np.minimum(tiles_array[:, 0] + tiles_array[:, 2], height) - tiles_array[:, 0]
    tiles_array[:, 3] = np.minimum(tiles_array[:, 1] + tiles_array[:, 3], width) - tiles_array[:, 1]

    return tiles_folder, tiles_array


def _bright_surface_attenuation(subtile_raw: np.ndarray) -> np.ndarray:
    """Compute bright surface attenuation for a single subtile.

    Reference: ``identify_bright_bare_surfaces`` (lines 1216-1239).
    Applied per-subtile BEFORE Gaussian mosaicking (line 1600).

    Args:
        subtile_raw: ``(T, SIZE+14, SIZE+14, 17)`` UN-normalized feature stack.

    Returns:
        ``(SIZE, SIZE)`` float32 attenuation mask in [0, 1].
    """
    from scipy.ndimage import binary_dilation, distance_transform_edt as distance

    # EVI from channels 0 (Blue), 2 (Red), 3 (NIR)
    blue = np.clip(subtile_raw[..., 0], 0, 1)
    red = np.clip(subtile_raw[..., 2], 0, 1)
    nir = np.clip(subtile_raw[..., 3], 0, 1)
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    evi = np.clip(evi, -1.5, 1.5)

    swir16 = np.clip(subtile_raw[..., 8], 0, 1)
    nir_swir = nir / (swir16 + 0.01)
    flag = nir_swir < 0.9
    flag = flag * (np.mean(subtile_raw[..., :3], axis=-1) > 0.2)
    flag = flag * (evi < 0.3)

    bright = np.sum(flag, axis=0) > 1
    bright = binary_dilation(1 - bright, iterations=2)
    bright = binary_dilation(1 - bright, iterations=1)

    blurred = distance(1 - bright).astype(np.float32)
    blurred[blurred > 3] = 3
    blurred = blurred / 3

    # Crop BORDER on each side to match prediction output size
    return blurred[BORDER:-BORDER, BORDER:-BORDER]


def mosaic_predictions(
    feature_stack: np.ndarray,
    predict_session,
    tile_size: int = SIZE,
    length: int = 4,
    gauss_sigma: int = 36,
) -> np.ndarray:
    """Run prediction on overlapping subtiles and mosaic with Gaussian blending.

    Matches the reference two-phase approach:
    1. Per-subtile: normalize → predict → bright_surface attenuate → save
    2. Mosaic: load all subtile predictions → Gaussian-weighted blend

    Args:
        feature_stack: ``(T, H, W, 17)`` full tile feature stack.
            T should be ``length + 1`` (4 quarterly + 1 median = 5).
        predict_session: :class:`PredictSession`.
        tile_size: Output subtile size (158 by default).
        length: Temporal sequence length (4 for quarterly).
        gauss_sigma: Sigma for Gaussian blending kernel.

    Returns:
        ``(H, W)`` uint8 tree cover [0-100] with 255 = nodata.
    """
    T, H, W, B = feature_stack.shape
    input_size = tile_size + 2 * BORDER  # 172

    tiles_folder, tiles_array = make_windows(H, W, tile_size=tile_size)
    n_windows = len(tiles_folder)

    logger.debug(f"Predicting {n_windows} overlapping windows ({tile_size}→{input_size})")

    # Phase 1: predict each subtile
    subtile_preds = []
    for i in range(n_windows):
        array = tiles_array[i]
        start_x, start_y = int(array[0]), int(array[1])
        ww, wh = int(array[2]), int(array[3])

        subtile = np.copy(feature_stack[:, start_x:start_x + ww, start_y:start_y + wh, :])

        # Pad edge/corner subtiles with reflection (reference lines 1497-1515)
        if subtile.shape[2] == tile_size + BORDER:
            pad_u = BORDER if start_y == 0 else 0
            pad_d = BORDER if start_y != 0 else 0
            subtile = np.pad(subtile, ((0, 0), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
        if subtile.shape[1] == tile_size + BORDER:
            pad_l = BORDER if start_x == 0 else 0
            pad_r = BORDER if start_x != 0 else 0
            subtile = np.pad(subtile, ((0, 0), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')

        # Zero-pad if still not full input size
        if subtile.shape[1] < input_size or subtile.shape[2] < input_size:
            padded = np.zeros((T, input_size, input_size, B), dtype=subtile.dtype)
            padded[:, :subtile.shape[1], :subtile.shape[2], :] = subtile
            subtile = padded

        # Compute bright surface attenuation BEFORE normalization
        bright_attn = _bright_surface_attenuation(subtile)

        # Normalize and predict
        subtile_norm = normalize_subtile(subtile)
        pred = predict_subtile(subtile_norm, predict_session, tile_size, length)

        # Apply bright surface attenuation (reference line 1600)
        if not np.all(pred == 255):
            pred = pred * bright_attn

        pred = np.around(pred, 3).astype(np.float32)
        subtile_preds.append(pred)

    # Phase 2: Gaussian-weighted mosaic (reference load_mosaic_predictions)
    accum = np.zeros((H, W), dtype=np.float64)
    weight_map = np.zeros((H, W), dtype=np.float64)
    gauss = fspecial_gauss(tile_size, gauss_sigma).astype(np.float64)

    for i in range(n_windows):
        folder = tiles_folder[i]
        folder_x, folder_y = int(folder[0]), int(folder[1])
        pred = subtile_preds[i]

        if np.all(pred == 255):
            continue

        ph, pw = pred.shape
        g = gauss[:ph, :pw].copy()
        # Exclude nodata from blending
        g[pred > 1.0] = 0.0

        accum[folder_x:folder_x + ph, folder_y:folder_y + pw] += pred * 100 * g
        weight_map[folder_x:folder_x + ph, folder_y:folder_y + pw] += g

    # Normalise
    valid = weight_map > 0
    result = np.full((H, W), 255.0, dtype=np.float64)
    result[valid] = accum[valid] / weight_map[valid]

    # Post-processing thresholds
    result[(result <= 15) & (result != 255)] = 0
    result[result > 100] = 255

    return result.astype(np.uint8)
