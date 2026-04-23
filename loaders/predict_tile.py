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

import hashlib
import json
import os
import tempfile
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from gri_tile_pipeline.phase_timer import PhaseTimer, timed


_MODEL_SHA256_CACHE: Dict[str, str] = {}

# Files the Dockerfile bakes the pipeline git sha into. Checked in order;
# first non-empty read wins. Kept in sync with `COPY docker/.git_sha ...`
# in docker/PredictDockerfile.
_GIT_SHA_FILES = ("/function/.git_sha", "/var/task/.git_sha")


def _resolve_container_git_sha() -> Optional[str]:
    """Return the pipeline git sha baked into the Lambda image, or None.

    Precedence: ``GRI_GIT_SHA`` env var, then the sha file the Dockerfile
    copies into ``/function/.git_sha`` (Lithops/Lambda layout). Returning
    None is fine — the STAC sidecar treats ``gri:git_sha`` as nullable.
    """
    env_sha = (os.environ.get("GRI_GIT_SHA") or "").strip()
    if env_sha:
        return env_sha
    for path in _GIT_SHA_FILES:
        try:
            with open(path) as f:
                sha = f.read().strip()
                if sha and sha != "unknown":
                    return sha
        except OSError:
            continue
    return None


def _compute_model_sha256(pb_path: str) -> Optional[str]:
    """Return SHA-256 of the frozen graph at *pb_path*, cached per-process."""
    cached = _MODEL_SHA256_CACHE.get(pb_path)
    if cached:
        return cached
    try:
        h = hashlib.sha256()
        with open(pb_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        sha = h.hexdigest()
        _MODEL_SHA256_CACHE[pb_path] = sha
        return sha
    except OSError:
        return None


def _build_provenance(
    *,
    model_dir: str,
    pipeline_version: Optional[str],
    git_sha: Optional[str],
    run_id: Optional[str],
) -> Dict[str, Any]:
    """Assemble provenance fields for the STAC sidecar.

    ``model_dir`` is the local directory holding ``predict_graph-172.pb``.
    We hash the .pb file once per worker (cached) so subsequent tiles in the
    same Lambda invocation skip the rehash.
    """
    from importlib.metadata import PackageNotFoundError, version

    pb_name = "predict_graph-172.pb"
    pb_path = os.path.join(model_dir, pb_name)
    model_sha = _compute_model_sha256(pb_path)

    if pipeline_version is None:
        try:
            pipeline_version = version("gri-tile-pipeline")
        except PackageNotFoundError:
            pipeline_version = "unknown"
    if git_sha is None:
        git_sha = _resolve_container_git_sha()

    return {
        "model": {
            "name": "predict_graph-172",
            "path": pb_path,
            "sha256": model_sha,
            "input_size": 172,
            "output_size": 158,
            "length": 4,
        },
        "pipeline_version": pipeline_version,
        "git_sha": git_sha,
        "run_id": run_id,
    }


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


def _cog_write(dst_path: str, arr: np.ndarray, lon: float, lat: float) -> None:
    """Write *arr* as a Cloud Optimized GeoTIFF at *dst_path*.

    COG driver handles internal tiling + overview generation automatically.
    Predictions are tree-cover % (uint8, 0-100) with nodata=255; ``average``
    resampling respects the nodata sentinel when building overviews.
    """
    import rasterio
    from rasterio.transform import from_bounds
    from gri_tile_pipeline.storage.stac import tile_bbox

    west, south, east, north = tile_bbox(lon, lat)
    h, w = arr.shape
    transform = from_bounds(west, south, east, north, w, h)

    with rasterio.open(
        dst_path,
        "w",
        driver="COG",
        height=h,
        width=w,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        nodata=255,
        compress="deflate",
        blocksize=256,
        overview_resampling="average",
    ) as dst:
        dst.write(arr, 1)


def _write_geotiff(store, key: str, arr: np.ndarray, lon: float, lat: float):
    """Write the prediction COG to *store* at *key*."""
    import obstore as obs

    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp.close()
    try:
        _cog_write(tmp.name, arr, lon, lat)
        with open(tmp.name, "rb") as f:
            obs.put(store, key, f.read())
    finally:
        os.remove(tmp.name)


def _write_geotiff_local(path: str, arr: np.ndarray, lon: float, lat: float):
    """Write the prediction COG directly to a local *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _cog_write(path, arr, lon, lat)


def _build_stac_item(
    *,
    x_tile: int,
    y_tile: int,
    year: int,
    lon: float,
    lat: float,
    asset_href: str,
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    from gri_tile_pipeline.storage.stac import build_predict_stac_item

    return build_predict_stac_item(
        x_tile=x_tile,
        y_tile=y_tile,
        year=year,
        lon=lon,
        lat=lat,
        asset_href=asset_href,
        model=provenance["model"],
        pipeline_version=provenance["pipeline_version"],
        git_sha=provenance["git_sha"],
        run_id=provenance["run_id"],
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _write_stac_sidecar(store, key: str, item: Dict[str, Any]) -> None:
    """Upload the STAC Item JSON next to the COG."""
    import obstore as obs

    data = json.dumps(item, indent=2, sort_keys=True).encode("utf-8")
    obs.put(store, key, data)


def _write_stac_sidecar_local(path: str, item: Dict[str, Any]) -> None:
    """Write the STAC Item JSON next to a local COG output."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(item, f, indent=2, sort_keys=True)


def _fill_zeros_with_temporal_median(arr: np.ndarray) -> np.ndarray:
    """Replace zero values with non-zero temporal median (reference helper).

    Matches reference fill_zeros_with_temporal_median: uses nanmedian
    after converting zeros to NaN, so the replacement median excludes
    zero values from the computation.
    """
    xf = arr.astype(np.float32, copy=False)
    xf_nz = np.where(xf == 0, np.nan, xf)
    median = np.nanmedian(xf_nz, axis=0)
    median = np.where(np.isnan(median), 0.0, median)
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
    clm: Optional[np.ndarray] = None,
    length: int = 4,
    enable_cloud_removal: bool = True,
    diagnostics: Optional[Dict] = None,
    intermediates: Optional[Dict] = None,
    seed: Optional[int] = None,
    timer: Optional["PhaseTimer"] = None,
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
        diagnostics: if provided, intermediate outputs are stored into this dict

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

    # Phase timing: _tick attributes elapsed time since the previous checkpoint
    # to a named phase. No-op if caller passed timer=None. Keeps per-phase
    # instrumentation low-diff without reindenting the phase bodies.
    _last = [time.perf_counter()]

    def _tick(name: str) -> None:
        now = time.perf_counter()
        if timer is not None:
            timer.record(name, now - _last[0])
        _last[0] = now

    # ------------------------------------------------------------------
    # 1. Convert to float32 [0, 1]
    # ------------------------------------------------------------------
    if s2_10.dtype == np.uint16:
        s2_10 = s2_10.astype(np.float32) / 65535.0
    if s2_20.dtype == np.uint16:
        s2_20 = s2_20.astype(np.float32) / 65535.0
    # Reference code only uses first 6 of the 20m bands (B05, B06, B07, B8A, B11, B12).
    # Some datasets include a 7th band (e.g. B09 at 60m) — discard it.
    if s2_20.shape[-1] > 6:
        s2_20 = s2_20[..., :6]
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
    # 1b. CLM preprocessing (reference lines 769-778)
    # ------------------------------------------------------------------
    if clm is not None:
        clm = clm.astype(np.float32)
        # Upsample 20m → 10m by nearest-neighbor repeat
        clm = clm.repeat(2, axis=1).repeat(2, axis=2)
        # Suppress Sen2Cor false positives: if 2 consecutive frames are cloudy, set to 0
        for i in range(clm.shape[0]):
            mins = max(i - 1, 0)
            maxs = min(i + 1, clm.shape[0])
            sums = np.sum(clm[mins:maxs], axis=0) == 2
            clm[mins:maxs, sums] = 0.0
        logger.info(f"CLM cloud %: {np.mean(clm, axis=(1, 2))}")

    _tick("preprocess_convert")

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
    # Bands 0-3 (B05, B06, B07, B8A) — simple bilinear upsample
    # Bands 4-5 (B11, B12 / SWIR) — "40m" bands: reference code applies 2×2 block-average
    #   before upsampling to match how these bands were originally at coarser resolution
    #   (reference lines 831-869)
    s2_20_up = np.zeros((T, H, W, s2_20.shape[-1]), dtype=np.float32)
    h20, w20 = s2_20.shape[1], s2_20.shape[2]
    for t in range(T):
        # Bands 0-3: standard bilinear
        for b in range(min(4, s2_20.shape[-1])):
            s2_20_up[t, :, :, b] = sk_resize(
                s2_20[t, :, :, b], (H, W), order=1, preserve_range=True
            )
        # Bands 4-5: 40m-aware upsampling (reference lines 831-869)
        for b in range(4, min(6, s2_20.shape[-1])):
            mid = s2_20[t, :, :, b]
            if h20 % 2 == 0 and w20 % 2 == 0:
                mid = mid.reshape(h20 // 2, 2, w20 // 2, 2).mean(axis=(1, 3))
                s2_20_up[t, :, :, b] = sk_resize(
                    mid, (H, W), order=1, preserve_range=True
                )
            elif h20 % 2 != 0 and w20 % 2 != 0:
                row0 = mid[0, :]
                col0 = mid[:, 0]
                inner = mid[1:, 1:].reshape(
                    (h20 - 1) // 2, 2, (w20 - 1) // 2, 2
                ).mean(axis=(1, 3))
                s2_20_up[t, 1:, 1:, b] = sk_resize(
                    inner, (H - 1, W - 1), order=1, preserve_range=True
                )
                s2_20_up[t, 0, :, b] = np.repeat(row0, 2)[:W]
                s2_20_up[t, :, 0, b] = np.repeat(col0, 2)[:H]
            elif h20 % 2 != 0:
                row0 = mid[0, :]
                inner = mid[1:].reshape(
                    (h20 - 1) // 2, 2, w20 // 2, 2
                ).mean(axis=(1, 3))
                s2_20_up[t, 1:, :, b] = sk_resize(
                    inner, (H - 1, W), order=1, preserve_range=True
                )
                s2_20_up[t, 0, :, b] = np.repeat(row0, 2)[:W]
            else:  # w20 % 2 != 0
                col0 = mid[:, 0]
                inner = mid[:, 1:].reshape(
                    h20 // 2, 2, (w20 - 1) // 2, 2
                ).mean(axis=(1, 3))
                s2_20_up[t, :, 1:, b] = sk_resize(
                    inner, (H, W - 1), order=1, preserve_range=True
                )
                s2_20_up[t, :, 0, b] = np.repeat(col0, 2)[:H]

    # Merge into 10-band array: [B2,B3,B4,B8, B5,B6,B7,B8A,B11,B12]
    s2_full = np.concatenate([s2_10, s2_20_up], axis=-1)

    if diagnostics is not None:
        diagnostics["s2_full_shape_post_merge"] = s2_full.shape
        diagnostics["s2_full_band_means_post_merge"] = s2_full.mean(axis=(0, 1, 2)).tolist()

    # Resize DEM to match
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

    if intermediates is not None:
        intermediates["s2_post_superres"] = s2_full.copy()

    _tick("spatial_align_superres")

    # ------------------------------------------------------------------
    # 3. Remove bad timesteps, handle missing values (BEFORE cloud removal)
    #    Reference: id_missing_px + interpolate_missing_vals run before
    #    identify_clouds_shadows (lines 873-922 in process_tile)
    # ------------------------------------------------------------------
    missing = id_missing_px(s2_full)
    if len(missing) > 0:
        s2_full = np.delete(s2_full, missing, axis=0)
        s2_dates = np.delete(s2_dates, missing)
        logger.info(f"Removed {len(missing)} bad timesteps, {s2_full.shape[0]} remain")

    # Fill any NaN values with temporal median
    s2_full = interpolate_na_vals(s2_full)

    # ------------------------------------------------------------------
    # 4. Cloud removal pipeline
    # ------------------------------------------------------------------
    interp_mask = None  # Will be set if cloud removal runs
    if enable_cloud_removal and s2_full.shape[0] >= 3:
        # identify_clouds_shadows needs raw DEM in meters
        dem_raw = dem  # dem is still in meters at this point
        cloud_probs, fcps = identify_clouds_shadows(s2_full, dem_raw)

        # Merge CLM after cloud detection (reference lines 929-934)
        # Note: reference tries clm[fcps]=0 but this always raises IndexError
        # (float array indexing), caught by try/except — so fcps filtering is
        # never actually applied. We match that behavior.
        if clm is not None:
            clm_use = clm[:s2_full.shape[0], :H, :W]
            cloud_probs = np.maximum(cloud_probs, clm_use)

        if intermediates is not None:
            intermediates["cloud_probs_initial"] = cloud_probs.copy()

        # Build soft interpolation masks
        interp_mask = id_areas_to_interp(cloud_probs)

        # Iterative pruning: remove images with >90% interp (up to 3 rounds)
        # Re-run cloud detection after each pruning round because the algorithm
        # is multi-temporal — removing heavily cloudy images improves detection
        # for remaining images (matching reference process_tile behavior).
        for _round in range(3):
            pct_interp = np.mean(interp_mask > 0, axis=(1, 2))
            to_drop = np.argwhere(pct_interp > 0.90).flatten()
            if len(to_drop) == 0 or s2_full.shape[0] - len(to_drop) < 3:
                break
            logger.info(f"Cloud pruning round {_round + 1}: dropping {len(to_drop)} images")
            keep = np.setdiff1d(np.arange(s2_full.shape[0]), to_drop)
            s2_full = s2_full[keep]
            s2_dates = s2_dates[keep]
            # Also prune CLM to match
            if clm is not None:
                clm = np.delete(clm, to_drop, axis=0)
            # Re-detect clouds on reduced image stack
            cloud_probs, fcps = identify_clouds_shadows(s2_full, dem_raw)
            # Re-merge CLM after re-detection (reference lines 959-965)
            if clm is not None:
                clm_use = clm[:s2_full.shape[0], :H, :W]
                cloud_probs = np.maximum(cloud_probs, clm_use)
            interp_mask = id_areas_to_interp(cloud_probs)

        if intermediates is not None:
            intermediates["cloud_probs_final"] = cloud_probs.copy()
            intermediates["interp_mask"] = interp_mask.copy()

        # Temporal interpolation: blend cloudy areas with cloud-free mosaic
        s2_full, _interp_out, cloud_to_remove = remove_cloud_and_shadows(
            s2_full, cloud_probs, fcps, seed=seed,
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

    if diagnostics is not None:
        diagnostics["n_timesteps_after_cloud"] = s2_full.shape[0]

    if intermediates is not None:
        intermediates["s2_post_cloud"] = s2_full.copy()

    _tick("cloud_removal")

    # Normalize DEM: reference line 1084 divides by 90 (meters → ~[0, 0.5])
    dem = dem / 90.0

    # Clip S2 to [0, 1] BEFORE zero/one replacement (reference line 1085)
    s2_full = np.clip(s2_full, 0, 1)

    # Second NaN fill on clipped data (reference line 1266)
    s2_full = interpolate_na_vals(s2_full)

    # ------------------------------------------------------------------
    # Compute median composite from pre-Whittaker data (reference lines 1269-1277)
    # This median is used for the 5th frame of the feature stack.
    # Reference computes it BEFORE deal_w_missing_px, from all post-cloud timesteps.
    # ------------------------------------------------------------------
    s2_median_raw = np.median(s2_full, axis=0).astype(np.float32)  # (H, W, 10)
    indices_from_raw = make_indices(s2_full)  # (T, H, W, 4)
    indices_median_raw = np.median(indices_from_raw, axis=0).astype(np.float32)  # (H, W, 4)

    # ------------------------------------------------------------------
    # deal_w_missing_px equivalent (reference lines 1148-1171)
    # Stricter missing pixel removal (threshold=10 vs initial 11)
    # ------------------------------------------------------------------
    missing_strict = id_missing_px(s2_full, 10)
    if len(missing_strict) > 0:
        s2_full = np.delete(s2_full, missing_strict, axis=0)
        s2_dates = np.delete(s2_dates, missing_strict)
        logger.info(f"Post-cloud id_missing_px(10): removed {len(missing_strict)} images, "
                     f"{s2_full.shape[0]} remain")

    # Replace zero and 1.0 values with temporal median (reference lines 1156-1164)
    # Reference recomputes np.median(arr, axis=0) inside the loop (cascading updates)
    if np.sum(s2_full == 0) > 0:
        for i in range(s2_full.shape[0]):
            arr_i = s2_full[i]
            zero_mask = arr_i == 0
            if np.any(zero_mask):
                arr_i[zero_mask] = np.median(s2_full, axis=0)[zero_mask]
    if np.sum(s2_full == 1) > 0:
        for i in range(s2_full.shape[0]):
            arr_i = s2_full[i]
            one_mask = arr_i == 1
            if np.any(one_mask):
                arr_i[one_mask] = np.median(s2_full, axis=0)[one_mask]

    # Remove timesteps with NaN values (reference lines 1165-1170)
    nan_timesteps = np.argwhere(
        np.sum(np.isnan(s2_full), axis=(1, 2, 3)) > 0
    ).flatten()
    if len(nan_timesteps) > 0:
        s2_full = np.delete(s2_full, nan_timesteps, axis=0)
        s2_dates = np.delete(s2_dates, nan_timesteps)
        logger.info(f"Removed {len(nan_timesteps)} NaN timesteps, {s2_full.shape[0]} remain")

    # ------------------------------------------------------------------
    # 4. Compute indices from raw post-deal_w_missing_px data, THEN resample
    #    Reference: make_and_smooth_indices computes indices from raw irregular
    #    timesteps, then resamples to 24 biweekly, then Whittaker smooths.
    #    Indices are nonlinear (EVI, MSAVI2) so order matters:
    #    f(resample(x)) != resample(f(x))
    # ------------------------------------------------------------------
    indices_irregular = make_indices(s2_full)  # (T_irregular, H, W, 4)

    # Resample both S2 and indices to 24 biweekly composites
    s2_full_24, max_gap = resample_to_biweekly(s2_full, s2_dates)
    indices_raw, _ = resample_to_biweekly(indices_irregular, s2_dates)  # (24, H, W, 4)
    logger.info(f"Resampled to {s2_full_24.shape[0]} biweekly composites (max gap: {max_gap}d)")

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

    if diagnostics is not None:
        diagnostics["s2_12_band_means"] = s2_12.mean(axis=(0, 1, 2)).tolist()
        diagnostics["indices_12_band_means"] = indices_12.mean(axis=(0, 1, 2)).tolist()

    _tick("resample_indices_whittaker")

    # ------------------------------------------------------------------
    # 7. Prepare S1: resize to match spatial dims, take first 12 frames
    # ------------------------------------------------------------------
    # Reference uses pad/crop (not bilinear resize) for S1 alignment:
    # 1. adjust_shape: edge-pad short dims, crop oversized dims
    # 2. pad_crop_to_hw: center-aligned final pad/crop with constant 0
    s1_use = s1[:12].astype(np.float32) if s1.ndim == 4 else s1[:12, :, :, np.newaxis]
    if s1_use.shape[1] != H or s1_use.shape[2] != W:
        h_s1, w_s1 = s1_use.shape[1], s1_use.shape[2]
        # Step 1: adjust_shape — edge-pad if too small, center-crop if too large
        if h_s1 < H:
            pad = (H - h_s1) // 2
            if pad == 0:
                s1_use = np.pad(s1_use, ((0, 0), (1, 0), (0, 0), (0, 0)), 'edge')
            else:
                s1_use = np.pad(s1_use, ((0, 0), (pad, pad), (0, 0), (0, 0)), 'edge')
        elif h_s1 > H:
            crop = (h_s1 - H) // 2
            if crop == 0:
                s1_use = s1_use[:, 1:, :, :]
            else:
                s1_use = s1_use[:, crop:-crop, :, :]
        if w_s1 < W:
            pad = (W - w_s1) // 2
            if pad == 0:
                s1_use = np.pad(s1_use, ((0, 0), (0, 0), (1, 0), (0, 0)), 'edge')
            else:
                s1_use = np.pad(s1_use, ((0, 0), (0, 0), (pad, pad), (0, 0)), 'edge')
        elif w_s1 > W:
            crop = (W - w_s1) // 2
            if crop == 0:
                s1_use = s1_use[:, :, 1:, :]
            else:
                s1_use = s1_use[:, :, crop:-crop, :]
        # Step 2: pad_crop_to_hw — center-aligned final alignment
        ch, cw = s1_use.shape[1], s1_use.shape[2]
        if ch != H or cw != W:
            hs = max(0, (ch - H) // 2)
            ws = max(0, (cw - W) // 2)
            he = min(ch, hs + H)
            we = min(cw, ws + W)
            s1_use = s1_use[:, hs:he, ws:we, :]
            ch2, cw2 = s1_use.shape[1], s1_use.shape[2]
            pad_h, pad_w = max(0, H - ch2), max(0, W - cw2)
            if pad_h > 0 or pad_w > 0:
                ph1, ph2 = pad_h // 2, pad_h - pad_h // 2
                pw1, pw2 = pad_w // 2, pad_w - pad_w // 2
                s1_use = np.pad(s1_use, ((0, 0), (ph1, ph2), (pw1, pw2), (0, 0)),
                                mode='constant', constant_values=0)

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

    _tick("s1_align_quarterly")

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
    # Reference computes median from pre-Whittaker, pre-deal_w_missing_px data
    # (lines 1269-1277, 1525-1527). s2_median_raw and indices_median_raw were
    # computed earlier from post-cloud, post-clip data before zero/one replacement.
    s1_median = np.median(s1_use, axis=0)      # (H, W, 2)

    feature_stack[-1, :, :, :10] = s2_median_raw
    feature_stack[-1, :, :, 11:13] = s1_median
    feature_stack[-1, :, :, 13:] = indices_median_raw

    logger.info(f"Feature stack: {feature_stack.shape}")

    if intermediates is not None:
        intermediates["feature_stack"] = feature_stack.copy()

    if diagnostics is not None:
        ch_stats = {}
        for ch in range(17):
            ch_data = feature_stack[:, :, :, ch]
            ch_stats[ch] = {
                "min": float(ch_data.min()),
                "max": float(ch_data.max()),
                "mean": float(ch_data.mean()),
            }
        diagnostics["feature_stack_channel_stats"] = ch_stats
        diagnostics["feature_stack_shape"] = feature_stack.shape

    _tick("feature_stack_assembly")

    # ------------------------------------------------------------------
    # 8. Predict with overlapping subtiles
    #    (bright surface attenuation is applied per-subtile inside mosaic_predictions)
    # ------------------------------------------------------------------
    predict_sess = load_predict_graph(model_path)
    predictions = mosaic_predictions(feature_stack, predict_sess, length=length, interp=interp_mask)

    if intermediates is not None:
        intermediates["prediction_raw"] = predictions.copy()

    _tick("tf_inference")

    # ------------------------------------------------------------------
    # 9. Post-process: nodata masking
    # ------------------------------------------------------------------
    predictions = apply_nodata_mask(predictions, s2_12)
    _tick("postprocess")

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
    seed: Optional[int] = None,
    prediction_key_override: Optional[str] = None,
    ard_keys_override: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Lithops entry-point for single-tile prediction.

    ``prediction_key_override`` / ``ard_keys_override`` are used by parity
    tooling to (a) avoid clobbering the production FINAL.tif and (b) read ARD
    from non-canonical paths. Production callers leave both as None.

    Returns a structured dict with ``status``, ``error_message``, etc.
    """
    import obstore as obs
    from gri_tile_pipeline.storage.obstore_utils import from_dest

    aws_profile = kwargs.get("aws_profile")

    tag = f"{X_tile}X{Y_tile}Y"
    logger.info(f"predict_tile: {tag} year={year}")

    phase_timer = PhaseTimer()
    wallclock_start = time.perf_counter()

    try:
        # Set up output store
        store = from_dest(dest, region="us-east-1", profile=aws_profile)

        # -------------------------------------------------------
        # 1. Load ARD data
        # -------------------------------------------------------
        if ard_keys_override:
            required = ("s2_10", "s2_20", "s1", "dem", "clouds", "s2_dates")
            missing = [s for s in required if s not in ard_keys_override]
            if missing:
                raise ValueError(
                    f"ard_keys_override missing sources: {missing}. "
                    f"Required: {required}"
                )
            ard_keys = ard_keys_override
            logger.info(f"Loading ARD for {tag} via override ({len(ard_keys)} keys)")
        else:
            base_key = f"{year}/raw/{X_tile}/{Y_tile}/raw"
            ard_keys = {
                "s2_10":    f"{base_key}/s2_10/{tag}.hkl",
                "s2_20":    f"{base_key}/s2_20/{tag}.hkl",
                "s1":       f"{base_key}/s1/{tag}.hkl",
                "dem":      f"{base_key}/misc/dem_{tag}.hkl",
                "clouds":   f"{base_key}/clouds/clouds_{tag}.hkl",
                "s2_dates": f"{base_key}/misc/s2_dates_{tag}.hkl",
            }
            logger.info(f"Loading ARD for {tag}")
        with timed(phase_timer, "s3_download_hkl"):
            s2_10    = _load_hkl(store, ard_keys["s2_10"])
            s2_20    = _load_hkl(store, ard_keys["s2_20"])
            s1       = _load_hkl(store, ard_keys["s1"])
            dem      = _load_hkl(store, ard_keys["dem"])
            clouds   = _load_hkl(store, ard_keys["clouds"])
            s2_dates = _load_hkl(store, ard_keys["s2_dates"])

        # -------------------------------------------------------
        # 2-8. Run prediction pipeline
        # -------------------------------------------------------
        if model_path is None:
            model_path = os.environ.get("MODEL_PATH", "/tmp/models")

        # Download model from S3 if needed
        if model_path.startswith("s3://"):
            with timed(phase_timer, "model_download"):
                model_store = from_dest(model_path, region="us-east-1", profile=aws_profile)
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
            seed=seed,
            timer=phase_timer,
        )

        # -------------------------------------------------------
        # 9. Write COG + STAC sidecar
        # -------------------------------------------------------
        if prediction_key_override:
            out_key = prediction_key_override
            logger.info(f"Writing prediction to override key: {out_key}")
        else:
            out_key = f"{year}/tiles/{X_tile}/{Y_tile}/{tag}_FINAL.tif"
        with timed(phase_timer, "cog_write"):
            _write_geotiff(store, out_key, predictions, lon, lat)
            logger.info(f"Wrote {out_key}")

        stac_key = out_key[:-len(".tif")] + ".json" if out_key.endswith(".tif") else out_key + ".json"
        provenance = _build_provenance(
            model_dir=model_path,
            pipeline_version=kwargs.get("pipeline_version"),
            git_sha=kwargs.get("git_sha"),
            run_id=kwargs.get("run_id"),
        )
        item = _build_stac_item(
            x_tile=X_tile, y_tile=Y_tile, year=year, lon=lon, lat=lat,
            asset_href=os.path.basename(out_key),
            provenance=provenance,
        )
        with timed(phase_timer, "stac_write"):
            _write_stac_sidecar(store, stac_key, item)
            logger.info(f"Wrote {stac_key}")

        return {
            "status": "success",
            "tile": tag,
            "year": year,
            "output_key": out_key,
            "stac_key": stac_key,
            "model_sha256": provenance["model"]["sha256"],
            "shape": list(predictions.shape),
            "phase_timings": phase_timer.as_dict(),
            "wallclock_sec": round(time.perf_counter() - wallclock_start, 4),
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
            "phase_timings": phase_timer.as_dict(),
            "wallclock_sec": round(time.perf_counter() - wallclock_start, 4),
        }


def run_local(
    ard_dir: str,
    model_path: str,
    output_path: str,
    lon: float = 0.0,
    lat: float = 0.0,
    *,
    x_tile: Optional[int] = None,
    y_tile: Optional[int] = None,
    year: Optional[int] = None,
    pipeline_version: Optional[str] = None,
    git_sha: Optional[str] = None,
    run_id: Optional[str] = None,
) -> np.ndarray:
    """Run prediction on local ARD files (for testing/validation).

    Args:
        ard_dir: Directory containing raw_v2/ structure with HKL files.
        model_path: Directory containing predict_graph-172.pb.
        output_path: Where to write the output COG.
        lon: Tile center longitude (for GeoTIFF metadata).
        lat: Tile center latitude (for GeoTIFF metadata).
        x_tile, y_tile, year: Tile identifiers for the STAC sidecar. When
            any is omitted, the sidecar is skipped (tests/parity callers
            that don't care about provenance pass lon/lat only).

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

    if x_tile is not None and y_tile is not None and year is not None:
        provenance = _build_provenance(
            model_dir=model_path,
            pipeline_version=pipeline_version,
            git_sha=git_sha,
            run_id=run_id,
        )
        item = _build_stac_item(
            x_tile=x_tile, y_tile=y_tile, year=year, lon=lon, lat=lat,
            asset_href=os.path.basename(output_path),
            provenance=provenance,
        )
        sidecar_path = (
            output_path[:-len(".tif")] + ".json"
            if output_path.endswith(".tif")
            else output_path + ".json"
        )
        _write_stac_sidecar_local(sidecar_path, item)
        logger.info(f"Wrote {sidecar_path}")

    return predictions
