"""Lambda worker for tree-cover prediction on a single tile.

Lithops entry-point: ``run(year, lon, lat, X_tile, Y_tile, dest, ...)``

Pipeline per tile (matching reference download_and_predict_job.py):
  1. Load ARD .hkl files from S3 (s2_10, s2_20, s1, dem, clouds, dates)
  2. Preprocess: cloud removal, super-resolution, Whittaker smoothing, indices
  3. Quarterly reduction: 12 monthly → 4 quarterly (median of 3-month groups)
  4. Build 5-frame × 17-channel feature stack (4 quarterly + 1 median)
  5. Predict: overlapping subtile windows with Gaussian blending
  6. Post-process: bright surface attenuation, nodata masking
  7. Write uint8 GeoTIFF to ``{year}/tiles/{X}/{Y}/{X}X{Y}Y_FINAL.tif``
"""

from __future__ import annotations

import os
import tempfile
import traceback
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger


def _load_hkl(store, key: str):
    """Download an HKL file from obstore and load it."""
    import hickle as hkl
    import obstore as obs

    data = obs.get(store, key)
    buf = data.bytes()
    tmp = tempfile.NamedTemporaryFile(suffix=".hkl", delete=False)
    tmp.write(buf)
    tmp.close()
    try:
        return hkl.load(tmp.name)
    finally:
        os.remove(tmp.name)


def _load_hkl_local(path: str):
    """Load an HKL file from local filesystem."""
    import hickle as hkl
    return hkl.load(path)


def _write_geotiff(store, key: str, arr: np.ndarray, lon: float, lat: float):
    """Write a uint8 GeoTIFF for the prediction output."""
    import rasterio
    import obstore as obs
    from rasterio.transform import from_bounds

    tile_deg = 1.0 / 18.0  # ~0.0556 degrees
    west = lon - tile_deg / 2
    south = lat - tile_deg / 2
    east = lon + tile_deg / 2
    north = lat + tile_deg / 2

    h, w = arr.shape
    transform = from_bounds(west, south, east, north, w, h)

    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp.close()
    try:
        with rasterio.open(
            tmp.name,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
            compress="deflate",
            nodata=255,
        ) as dst:
            dst.write(arr, 1)

        with open(tmp.name, "rb") as f:
            obs.put(store, key, f.read())
    finally:
        os.remove(tmp.name)


def _write_geotiff_local(path: str, arr: np.ndarray, lon: float, lat: float):
    """Write a uint8 GeoTIFF directly to a local path."""
    import rasterio
    from rasterio.transform import from_bounds

    tile_deg = 1.0 / 18.0
    west = lon - tile_deg / 2
    south = lat - tile_deg / 2
    east = lon + tile_deg / 2
    north = lat + tile_deg / 2

    h, w = arr.shape
    transform = from_bounds(west, south, east, north, w, h)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        compress="deflate",
        nodata=255,
    ) as dst:
        dst.write(arr, 1)


def _fill_zeros_with_temporal_median(arr: np.ndarray) -> np.ndarray:
    """Replace zero values with temporal median (reference helper)."""
    median = np.median(arr, axis=0)
    for t in range(arr.shape[0]):
        zero_mask = arr[t] == 0
        if np.any(zero_mask):
            arr[t][zero_mask] = median[zero_mask]
    return arr


def predict_tile_from_arrays(
    s2_10: np.ndarray,
    s2_20: np.ndarray,
    s1: np.ndarray,
    dem: np.ndarray,
    clouds: np.ndarray,
    s2_dates: np.ndarray,
    model_path: str,
    length: int = 4,
    enable_cloud_removal: bool = True,
) -> np.ndarray:
    """Run the full prediction pipeline on pre-loaded arrays.

    This is the core logic, separated from I/O for testability.

    Args:
        s2_10: (T, H, W, 4) uint16 or float32 — 10m S2 bands
        s2_20: (T, H2, W2, 6) uint16 or float32 — 20m S2 bands
        s1: (T_s1, H_s1, W_s1, 2) uint16 or float32 — S1 VV, VH
        dem: (H, W) float32 — DEM elevation
        clouds: (T, H, W) bool or similar — cloud mask
        s2_dates: array of dates
        model_path: path to directory containing predict_graph-172.pb
        length: temporal sequence length (4 = quarterly, default)
        enable_cloud_removal: run multi-temporal cloud removal (default True)

    Returns:
        (H, W) uint8 prediction [0-100], 255=nodata
    """
    from skimage.transform import resize as sk_resize

    from gri_tile_pipeline.preprocessing.cloud_removal import (
        id_missing_px,
        interpolate_na_vals,
        identify_clouds_shadows,
        id_areas_to_interp,
        remove_cloud_and_shadows,
    )
    from gri_tile_pipeline.preprocessing.super_resolution import superresolve_tile
    from gri_tile_pipeline.preprocessing.whittaker import WhittakerSmoother
    from gri_tile_pipeline.preprocessing.indices import make_indices
    from gri_tile_pipeline.preprocessing.temporal_resampling import resample_to_biweekly
    from gri_tile_pipeline.inference.frozen_graph import load_predict_graph, load_superresolve_graph
    from gri_tile_pipeline.inference.subtile_predict import mosaic_predictions
    from gri_tile_pipeline.inference.postprocessing import apply_nodata_mask

    # ------------------------------------------------------------------
    # 1. Convert to float32 [0, 1]
    # ------------------------------------------------------------------
    if s2_10.dtype == np.uint16:
        s2_10 = s2_10.astype(np.float32) / 65535.0
    if s2_20.dtype == np.uint16:
        s2_20 = s2_20.astype(np.float32) / 65535.0
    if s1.dtype == np.uint16:
        s1 = s1.astype(np.float32) / 65535.0

    # Replace saturated S1 values (==1.0) with median (reference lines 785-788)
    for i in range(s1.shape[0]):
        for b in range(s1.shape[-1]):
            sat_mask = s1[i, :, :, b] == 1.0
            if np.any(sat_mask):
                med = np.median(s1[i, :, :, b][~sat_mask])
                s1[i, :, :, b][sat_mask] = med

    # Convert S1 linear backscatter to dB (reference lines 790-791)
    def _convert_to_db(x, min_db=22):
        x = 10 * np.log10(x + 1 / 65535)
        x[x < -min_db] = -min_db
        x = (x + min_db) / min_db
        return np.clip(x, 0, 1)

    for band in range(s1.shape[-1]):
        s1[..., band] = _convert_to_db(s1[..., band])

    if dem.dtype != np.float32:
        dem = dem.astype(np.float32)
    from scipy.ndimage import median_filter
    dem = median_filter(dem, size=5)  # reference line 797
    # NOTE: DEM normalization (/90) is deferred until after cloud removal,
    # which needs raw meters for elevation thresholds (e.g. dem >= 25m).

    # ------------------------------------------------------------------
    # 2. Align spatial dimensions: target = 2 × s2_20 shape (reference lines 800-806)
    # ------------------------------------------------------------------
    T = s2_10.shape[0]
    H = s2_20.shape[1] * 2  # target height (rows)
    W = s2_20.shape[2] * 2  # target width (cols)

    logger.info(f"ARD loaded: s2_10={s2_10.shape}, s2_20={s2_20.shape}, "
                f"s1={s1.shape}, dem={dem.shape}, target=({H},{W})")

    # Resize s2_10 to target dims if needed
    if s2_10.shape[1] != H or s2_10.shape[2] != W:
        s2_10_r = np.zeros((T, H, W, s2_10.shape[-1]), dtype=np.float32)
        for t in range(T):
            for b in range(s2_10.shape[-1]):
                s2_10_r[t, :, :, b] = sk_resize(
                    s2_10[t, :, :, b], (H, W), order=1, preserve_range=True
                )
        s2_10 = s2_10_r

    # Upsample 20m bands to target dims
    s2_20_up = np.zeros((T, H, W, s2_20.shape[-1]), dtype=np.float32)
    for t in range(T):
        for b in range(s2_20.shape[-1]):
            s2_20_up[t, :, :, b] = sk_resize(
                s2_20[t, :, :, b], (H, W), order=1, preserve_range=True
            )

    # Merge into 10-band array: [B2,B3,B4,B8, B5,B6,B7,B8A,B11,B12]
    s2_full = np.concatenate([s2_10, s2_20_up], axis=-1)

    # Resize DEM to match
    if dem.shape[0] != H or dem.shape[1] != W:
        dem = sk_resize(dem, (H, W), order=1, preserve_range=True).astype(np.float32)

    # Super-resolution (CNN-based if model file available, else bilinear fallback)
    try:
        sr_sess = load_superresolve_graph(model_path)
        s2_full = superresolve_tile(
            s2_full,
            sess=sr_sess.sess,
            sr_logits=sr_sess.logits,
            sr_inp=sr_sess.inp,
            sr_inp_bilinear=sr_sess.inp_bilinear,
        )
        logger.info("Applied CNN super-resolution")
    except (FileNotFoundError, OSError):
        logger.warning("superresolve_graph.pb not found — using bilinear upsampling only")
        s2_full = superresolve_tile(s2_full, sess=None)

    # ------------------------------------------------------------------
    # 3. Cloud removal pipeline
    # ------------------------------------------------------------------
    if enable_cloud_removal and s2_full.shape[0] >= 3:
        # identify_clouds_shadows needs raw DEM in meters
        dem_raw = dem  # dem is still in meters at this point
        cloud_probs, fcps = identify_clouds_shadows(s2_full, dem_raw)

        # Build soft interpolation masks
        interp_mask = id_areas_to_interp(cloud_probs)

        # Iterative pruning: remove images with >90% interp (up to 3 rounds)
        for _round in range(3):
            pct_interp = np.mean(interp_mask > 0.5, axis=(1, 2))
            to_drop = np.argwhere(pct_interp > 0.90).flatten()
            if len(to_drop) == 0 or s2_full.shape[0] - len(to_drop) < 3:
                break
            logger.info(f"Cloud pruning round {_round + 1}: dropping {len(to_drop)} images")
            keep = np.setdiff1d(np.arange(s2_full.shape[0]), to_drop)
            s2_full = s2_full[keep]
            cloud_probs = cloud_probs[keep]
            interp_mask = interp_mask[keep]
            fcps = fcps[keep]
            s2_dates = s2_dates[keep]
            # Recompute interp mask after pruning
            interp_mask = id_areas_to_interp(cloud_probs)

        # Temporal interpolation: blend cloudy areas with cloud-free mosaic
        s2_full, _interp_out, cloud_to_remove = remove_cloud_and_shadows(
            s2_full, cloud_probs, fcps,
        )
        if cloud_to_remove:
            unique_remove = sorted(set(cloud_to_remove))
            keep = np.setdiff1d(np.arange(s2_full.shape[0]), unique_remove)
            if len(keep) >= 3:
                s2_full = s2_full[keep]
                s2_dates = s2_dates[keep]
                logger.info(f"Removed {len(unique_remove)} fully-cloudy images after blending")
        logger.info(f"Cloud removal done: {s2_full.shape[0]} images remain")
    elif enable_cloud_removal:
        logger.warning("Too few images for cloud removal (<3), skipping")

    # Normalize DEM: reference line 1084 divides by 90 (meters → ~[0, 0.5])
    dem = dem / 90.0

    # ------------------------------------------------------------------
    # 4. Remove bad timesteps, handle missing values
    # ------------------------------------------------------------------
    missing = id_missing_px(s2_full)
    if len(missing) > 0:
        s2_full = np.delete(s2_full, missing, axis=0)
        logger.info(f"Removed {len(missing)} bad timesteps, {s2_full.shape[0]} remain")

    # Fill any NaN values with temporal median
    s2_full = interpolate_na_vals(s2_full)

    # Replace zero and 1.0 values with temporal median (reference lines 1156-1164)
    median_vals = np.median(s2_full, axis=0)
    for i in range(s2_full.shape[0]):
        zero_mask = s2_full[i] == 0
        if np.any(zero_mask):
            s2_full[i][zero_mask] = median_vals[zero_mask]
        one_mask = s2_full[i] == 1
        if np.any(one_mask):
            s2_full[i][one_mask] = median_vals[one_mask]

    s2_full = np.clip(s2_full, 0, 1)  # reference line 1085

    # ------------------------------------------------------------------
    # 4. Resample to 24 biweekly composites (reference: calculate_and_save_best_images)
    #    This ensures enough timesteps for Whittaker smoothing (24→12)
    # ------------------------------------------------------------------
    s2_full_24, max_gap = resample_to_biweekly(s2_full, s2_dates)
    logger.info(f"Resampled to {s2_full_24.shape[0]} biweekly composites (max gap: {max_gap}d)")

    # ------------------------------------------------------------------
    # 5. Compute vegetation indices BEFORE Whittaker smoothing
    #    (reference: indices are computed from resampled S2, then smoothed separately)
    # ------------------------------------------------------------------
    indices_raw = make_indices(s2_full_24)  # (24, H, W, 4)

    # Smooth indices separately (reference: lmbd=100, size=24, outsize=12)
    sm_indices = WhittakerSmoother(
        lmbd=100.0,
        size=24,
        nbands=4,
        dimx=H,
        dimy=W,
        outsize=12,
    )
    indices_12 = sm_indices.interpolate_array(indices_raw)  # (12, H, W, 4)

    # ------------------------------------------------------------------
    # 6. Whittaker smoothing of S2 bands (reference: lmbd=100, size=24, outsize=12)
    # ------------------------------------------------------------------
    n_s2_bands = s2_full_24.shape[-1]  # 10
    smoother = WhittakerSmoother(
        lmbd=100.0,
        size=24,
        nbands=n_s2_bands,
        dimx=H,
        dimy=W,
        outsize=12,
    )
    s2_12 = smoother.interpolate_array(s2_full_24)  # (12, H, W, 10)

    # ------------------------------------------------------------------
    # 7. Prepare S1: resize to match spatial dims, take first 12 frames
    # ------------------------------------------------------------------
    if s1.shape[1] != H or s1.shape[2] != W:
        s1_resized = np.zeros((min(s1.shape[0], 12), H, W, 2), dtype=np.float32)
        for t in range(s1_resized.shape[0]):
            for b in range(2):
                s1_resized[t, :, :, b] = sk_resize(
                    s1[t, :, :, b], (H, W), order=1, preserve_range=True
                )
        s1_use = s1_resized
    else:
        s1_use = s1[:12].astype(np.float32) if s1.ndim == 4 else s1[:12, :, :, np.newaxis]

    # Fill zero S1 values with temporal median
    s1_use = _fill_zeros_with_temporal_median(s1_use)

    # Pad S1 to 12 frames if needed
    if s1_use.shape[0] < 12:
        pad_t = 12 - s1_use.shape[0]
        s1_use = np.pad(s1_use, ((0, pad_t), (0, 0), (0, 0), (0, 0)), 'edge')

    # ------------------------------------------------------------------
    # 8. Quarterly reduction: 12 monthly → 4 quarterly via median
    #    (reference lines 1394-1398: reshape (12,...) → (4,3,...), median axis=1)
    # ------------------------------------------------------------------
    s2_q = np.median(s2_12.reshape(4, 3, H, W, 10), axis=1)        # (4, H, W, 10)
    indices_q = np.median(indices_12.reshape(4, 3, H, W, 4), axis=1)  # (4, H, W, 4)
    s1_q = np.median(s1_use.reshape(4, 3, H, W, 2), axis=1)        # (4, H, W, 2)

    logger.info(f"Quarterly reduction: S2 {s2_12.shape}→{s2_q.shape}, "
                f"S1 {s1_use.shape}→{s1_q.shape}")

    # ------------------------------------------------------------------
    # 9. Build 17-channel × 5-frame feature stack
    #    Reference: (length+1, H, W, 17) = (5, H, W, 17)
    #    Channels: 0-9 S2, 10 DEM, 11-12 S1, 13-16 indices
    #    Frames 0..3: quarterly smoothed data
    #    Frame 4: median composite
    # ------------------------------------------------------------------
    n_frames = length + 1  # 5
    feature_stack = np.zeros((n_frames, H, W, 17), dtype=np.float32)

    # Quarterly frames (0..3)
    feature_stack[:length, :, :, :10] = s2_q             # S2 10 bands
    feature_stack[:length, :, :, 11:13] = s1_q           # S1 VV, VH
    feature_stack[:length, :, :, 13:] = indices_q         # 4 indices

    # DEM in ALL frames (reference line 1524)
    feature_stack[:, :, :, 10] = np.broadcast_to(dem, (n_frames, H, W))

    # Median composite frame (last frame = index 4)
    # Reference uses median of pre-Whittaker data; we approximate with
    # median of 12 monthly smoothed data (closer to the full time series)
    s2_median = np.median(s2_12, axis=0)       # (H, W, 10)
    s1_median = np.median(s1_use, axis=0)      # (H, W, 2)
    indices_median = np.median(indices_12, axis=0)  # (H, W, 4)

    feature_stack[-1, :, :, :10] = s2_median
    feature_stack[-1, :, :, 11:13] = s1_median
    feature_stack[-1, :, :, 13:] = indices_median

    logger.info(f"Feature stack: {feature_stack.shape}")

    # ------------------------------------------------------------------
    # 8. Predict with overlapping subtiles
    #    (bright surface attenuation is applied per-subtile inside mosaic_predictions)
    # ------------------------------------------------------------------
    predict_sess = load_predict_graph(model_path)
    predictions = mosaic_predictions(feature_stack, predict_sess, length=length)

    # ------------------------------------------------------------------
    # 9. Post-process: nodata masking
    # ------------------------------------------------------------------
    predictions = apply_nodata_mask(predictions, s2_12)

    return predictions


def run(
    year: int,
    lon: float,
    lat: float,
    X_tile: int,
    Y_tile: int,
    dest: str,
    model_path: Optional[str] = None,
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Lithops entry-point for single-tile prediction.

    Returns a structured dict with ``status``, ``error_message``, etc.
    """
    import obstore as obs
    from obstore.store import LocalStore, from_url

    tag = f"{X_tile}X{Y_tile}Y"
    logger.info(f"predict_tile: {tag} year={year}")

    try:
        # Set up output store
        if dest.startswith("s3://"):
            store = from_url(dest, region="us-west-2")
        else:
            os.makedirs(dest, exist_ok=True)
            store = LocalStore(prefix=dest)

        base_key = f"{year}/raw/{X_tile}/{Y_tile}/raw"

        # -------------------------------------------------------
        # 1. Load ARD data
        # -------------------------------------------------------
        logger.info(f"Loading ARD for {tag}")
        s2_10 = _load_hkl(store, f"{base_key}/s2_10/{tag}.hkl")
        s2_20 = _load_hkl(store, f"{base_key}/s2_20/{tag}.hkl")
        s1 = _load_hkl(store, f"{base_key}/s1/{tag}.hkl")
        dem = _load_hkl(store, f"{base_key}/misc/dem_{tag}.hkl")
        clouds = _load_hkl(store, f"{base_key}/clouds/clouds_{tag}.hkl")
        s2_dates = _load_hkl(store, f"{base_key}/misc/s2_dates_{tag}.hkl")

        # -------------------------------------------------------
        # 2-8. Run prediction pipeline
        # -------------------------------------------------------
        if model_path is None:
            model_path = os.environ.get("MODEL_PATH", "/tmp/models")

        # Download model from S3 if needed
        if model_path.startswith("s3://"):
            model_store = from_url(model_path, region="us-west-2")
            local_model = "/tmp/models"
            os.makedirs(local_model, exist_ok=True)
            for obj in obs.list(model_store):
                for meta in obj:
                    key = meta["path"]
                    data = obs.get(model_store, key).bytes()
                    local_path = os.path.join(local_model, os.path.basename(key))
                    with open(local_path, "wb") as f:
                        f.write(data)
            model_path = local_model

        predictions = predict_tile_from_arrays(
            s2_10, s2_20, s1, dem, clouds, s2_dates, model_path,
        )

        # -------------------------------------------------------
        # 9. Write GeoTIFF
        # -------------------------------------------------------
        out_key = f"{year}/tiles/{X_tile}/{Y_tile}/{tag}_FINAL.tif"
        _write_geotiff(store, out_key, predictions, lon, lat)
        logger.info(f"Wrote {out_key}")

        return {
            "status": "success",
            "tile": tag,
            "year": year,
            "output_key": out_key,
            "shape": list(predictions.shape),
        }

    except Exception as e:
        logger.error(f"predict_tile failed for {tag}: {e}")
        return {
            "status": "failed",
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
            "error_type": "permanent",
            "tile": tag,
            "year": year,
        }


def run_local(
    ard_dir: str,
    model_path: str,
    output_path: str,
    lon: float = 0.0,
    lat: float = 0.0,
) -> np.ndarray:
    """Run prediction on local ARD files (for testing/validation).

    Args:
        ard_dir: Directory containing raw_v2/ structure with HKL files.
        model_path: Directory containing predict_graph-172.pb.
        output_path: Where to write the output GeoTIFF.
        lon: Tile center longitude (for GeoTIFF metadata).
        lat: Tile center latitude (for GeoTIFF metadata).

    Returns:
        (H, W) uint8 prediction array.
    """
    import hickle as hkl
    import glob

    # Find HKL files
    s2_10 = hkl.load(glob.glob(f"{ard_dir}/s2_10/*.hkl")[0])
    s2_20 = hkl.load(glob.glob(f"{ard_dir}/s2_20/*.hkl")[0])
    s1 = hkl.load(glob.glob(f"{ard_dir}/s1/*.hkl")[0])
    dem = hkl.load(glob.glob(f"{ard_dir}/misc/dem_*.hkl")[0])
    clouds = hkl.load(glob.glob(f"{ard_dir}/clouds/clouds_*.hkl")[0])
    s2_dates = hkl.load(glob.glob(f"{ard_dir}/misc/s2_dates_*.hkl")[0])

    logger.info(f"Loaded ARD: s2_10={s2_10.shape}, s2_20={s2_20.shape}, "
                f"s1={s1.shape}, dem={dem.shape}")

    predictions = predict_tile_from_arrays(
        s2_10, s2_20, s1, dem, clouds, s2_dates, model_path,
    )

    _write_geotiff_local(output_path, predictions, lon, lat)
    logger.info(f"Wrote {output_path} — shape={predictions.shape}, "
                f"mean={predictions[predictions < 255].mean():.1f}")

    return predictions
