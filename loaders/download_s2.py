#!/usr/bin/env python3
"""
Sentinel-2 L2A STAC-based downloader with legacy-compatible outputs.
Replicates the two-stage workflow from the legacy system.

Produces:
  - raw/clouds/{clouds,cloudmask,clean_steps}_{X}X{Y}Y.hkl
  - raw/misc/s2_dates_{X}X{Y}Y.hkl
  - raw/s2_10/{X}X{Y}Y.hkl  (uint16, (T,H10,W10,4))
  - raw/s2_20/{X}X{Y}Y.hkl  (uint16, (T,H20,W20,6))
"""

import os
import sys
import time
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path

import numpy as np
import xarray as xr
import click
from loguru import logger
import hickle as hkl
import random

import obstore as obs
from obstore.store import S3Store, LocalStore, from_url

import boto3
from pystac_client import Client
from odc.stac import load as stac_load, configure_rio
from shapely.geometry import box as shapely_box

# ----------------------------
# Configuration
# ----------------------------
EARTH_SEARCH_V1 = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"
S2_REGION = "us-west-2"

# Band definitions - matching legacy exactly
ASSETS_10M = ["blue", "green", "red", "nir"]  # B02, B03, B04, B08
ASSETS_20M = ["rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"]  # B05, B06, B07, B8A, B11, B12
SCL_ASSET = ["scl"]

# Cloud thresholds - matching legacy
CLOUD_HARD_DROP = 0.50      # >50% global clouds → drop
CLOUD_FINAL_MAX = 0.40      # Final threshold after local weighting

# SCL cloudy values - matching reference script
SCL_CLOUDY = {0, 1, 2, 3, 7, 8, 9, 10, 11}  # All problematic pixels

# ----------------------------
# Utilities
# ----------------------------
def _elapsed_ms(t_start: float) -> float:
    """Calculate elapsed time in milliseconds."""
    return (time.perf_counter() - t_start) * 1000.0

def _to_numpy(x) -> np.ndarray:
    """Robust conversion to NumPy array across xarray/dask versions.
    Prefers .to_numpy() (xarray) and falls back to np.asarray.
    """
    try:
        return x.to_numpy()
    except AttributeError:
        return np.asarray(x)

def to_uint16(arr01: np.ndarray) -> np.ndarray:
    """Quantize float [0,1] → uint16 (0..65535)."""
    return np.clip(np.round(arr01 * 65535.0), 0, 65535).astype(np.uint16)

def make_bbox(initial_bbx: list, expansion: int = 10) -> list:
    """
    Makes a (min_x, min_y, max_x, max_y) bounding box that
    is 2 * expansion 300 x 300 meter ESA LULC pixels
    """
    multiplier = 1/360  # ~300m in decimal degrees
    bbx = initial_bbx.copy()
    bbx[0] -= expansion * multiplier
    bbx[1] -= expansion * multiplier
    bbx[2] += expansion * multiplier
    bbx[3] += expansion * multiplier
    return bbx

def bbox2geojson(bbox: list) -> dict:
    """Convert a bounding box to a GeoJSON polygon."""
    coords = [
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]],
        [bbox[0], bbox[1]]
    ]
    return {"type": "Polygon", "coordinates": [coords]}

def extract_dates_legacy(date_dict: list, year: int) -> List[int]:
    """
    Legacy-compatible date extraction to julian days.
    Matches the original extract_dates function.
    """
    dates = []
    days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
    starting_days = np.cumsum(days_per_month)
    
    for date in date_dict:
        if hasattr(date, 'year'):
            # datetime object
            dates.append(((date.year - year) * 365) +
                        starting_days[(date.month - 1)] + date.day)
        else:
            # numpy datetime64
            dt_str = str(date)[:10]  # YYYY-MM-DD
            y, m, d = int(dt_str[:4]), int(dt_str[5:7]), int(dt_str[8:10])
            dates.append(((y - year) * 365) + starting_days[m - 1] + d)
    
    return dates

def to_doy(dates: np.ndarray, year: int) -> np.ndarray:
    """Convert dates to day-of-year (1-366)."""
    doy_list = []
    for date in dates:
        if isinstance(date, np.datetime64):
            dt = date
        else:
            dt = np.datetime64(date, 'D')
        
        dt_str = str(dt)[:10]
        y = int(dt_str[:4])
        if y == year:
            start = np.datetime64(f"{y}-01-01", 'D')
            doy = int((dt - start) / np.timedelta64(1, 'D')) + 1
            doy_list.append(doy)
    
    return np.array(doy_list, dtype=np.int64)

def remove_noise_clouds(arr: np.ndarray) -> np.ndarray:
    """
    Remove noise from cloud masks - legacy function.
    """
    arr = arr.copy()
    for t in range(arr.shape[0]):
        for x in range(1, arr.shape[1] - 1):
            for y in range(1, arr.shape[2] - 1):
                window = arr[t, x - 1:x + 2, y - 1:y + 2]
                if window[1, 1] > 0:
                    if np.sum(window > 0) <= 1 and np.sum(arr[:, x, y]) > arr.shape[0] - 1:
                        window = 0.
                        arr[t, x - 1:x + 2, y - 1:y + 2] = window
    return arr

def _check_for_alt_img(local_clouds: np.ndarray, cloud_dates: np.ndarray, 
                       current_date: int) -> bool:
    """Check if there's a better image within the same month."""
    # Simplified version - just check if significantly better exists
    month_mask = np.abs(cloud_dates - current_date) <= 30
    if np.sum(month_mask) <= 1:
        return False
    
    current_idx = np.where(cloud_dates == current_date)[0]
    if len(current_idx) == 0:
        return False
    
    current_cloud = local_clouds[current_idx[0]]
    other_clouds = local_clouds[month_mask & (cloud_dates != current_date)]
    
    return np.any(other_clouds < (current_cloud - 0.20))

# ----------------------------
# Storage utilities
# ----------------------------
def obstore_put_hkl(store: Union[S3Store, LocalStore], relpath: str, obj) -> None:
    """Save object as hickle file to obstore."""
    os.makedirs("/tmp", exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".hkl", delete=False)
    tmp.close()
    try:
        hkl.dump(obj, tmp.name, mode="w", compression="gzip")
        with open(tmp.name, "rb") as f:
            data = f.read()
        obs.put(store, relpath, data)
        logger.debug(f"Saved {relpath} ({len(data)/1024:.1f} KB)")
    finally:
        try:
            os.remove(tmp.name)
        except:
            pass

def ensure_dirs(store: Union[S3Store, LocalStore], *dirs: str) -> None:
    """Ensure directories exist in storage."""
    for d in dirs:
        try:
            if isinstance(store, LocalStore):
                Path(store.prefix, d).mkdir(parents=True, exist_ok=True)
            else:
                obs.put(store, d.rstrip("/") + "/.keep", b"")
        except:
            pass

@dataclass
class SavePaths:
    """Path structure for saving outputs."""
    root: str
    year: int
    X_tile: int
    Y_tile: int

    @property
    def base(self) -> str:
        return f"{self.year}/raw/{self.X_tile}/{self.Y_tile}/raw"

    @property
    def clouds_dir(self) -> str:
        return f"{self.base}/clouds"

    @property
    def misc_dir(self) -> str:
        return f"{self.base}/misc"

    @property
    def s2_10_dir(self) -> str:
        return f"{self.base}/s2_10"

    @property
    def s2_20_dir(self) -> str:
        return f"{self.base}/s2_20"

    def f_clouds(self) -> str:
        return f"{self.clouds_dir}/clouds_{self.X_tile}X{self.Y_tile}Y.hkl"

    def f_cloudmask(self) -> str:
        return f"{self.clouds_dir}/cloudmask_{self.X_tile}X{self.Y_tile}Y.hkl"

    def f_clean(self) -> str:
        return f"{self.clouds_dir}/clean_steps_{self.X_tile}X{self.Y_tile}Y.hkl"

    def f_s2_dates(self) -> str:
        return f"{self.misc_dir}/s2_dates_{self.X_tile}X{self.Y_tile}Y.hkl"

    def f_s2_10(self) -> str:
        return f"{self.s2_10_dir}/{self.X_tile}X{self.Y_tile}Y.hkl"

    def f_s2_20(self) -> str:
        return f"{self.s2_20_dir}/{self.X_tile}X{self.Y_tile}Y.hkl"

# ----------------------------
# Stage 1: Cloud Identification (replaces identify_clouds_big_bbx)
# ----------------------------
def identify_clouds_stac(items: List, bbox: list, year: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 1: Identify clean dates from cloud analysis.
    Replaces legacy identify_clouds_big_bbx function.
    
    Returns: (clouds_legacy, cloud_percent, clean_steps, local_clouds)
    """
    if len(items) == 0:
        return (np.zeros((0,0,0), np.float32), np.array([]), np.array([]), np.array([]))
    
    t_load = time.perf_counter()
    
    # Load SCL at 640m resolution (matching legacy)
    ds_scl = stac_load(
        items,
        bands=SCL_ASSET,
        bbox=bbox,
        resolution=640,  # meters, odc.stac handles conversion
        groupby="solar_day",
        resampling="nearest",
        dtype="uint8",
        chunks={}
    )
    
    logger.info(f"Loaded SCL for cloud analysis in {_elapsed_ms(t_load):.0f} ms")
    logger.debug(f"SCL shape: {ds_scl.sizes}")
    
    # Extract arrays (robust across xarray versions)
    scl = _to_numpy(ds_scl["scl"])  # (T,H,W)
    times = _to_numpy(ds_scl["time"])  # (T,)
        
    # Create cloud mask using legacy SCL values
    cloud_mask = np.zeros_like(scl, dtype=bool)
    for code in SCL_CLOUDY:
        cloud_mask |= (scl == code)
    
    # Handle invalid pixels
    valid = ~np.isnan(scl.astype(float))
    
    # Calculate cloud fraction
    cloud_frac = np.zeros_like(scl, dtype=np.float32)
    cloud_frac[valid] = cloud_mask[valid].astype(np.float32)
    cloud_frac[~valid] = np.nan
    
    T, H, W = cloud_frac.shape
    
    # Global cloud percentage
    cloud_percent = np.nanmean(cloud_frac, axis=(1, 2))
    
    # Local cloud assessment (center 30x30 pixels)
    cx, cy = H // 2, W // 2
    x0, x1 = max(0, cx - 15), min(H, cx + 15)
    y0, y1 = max(0, cy - 15), min(W, cy + 15)
    
    if x1 > x0 and y1 > y0:
        local_clouds = np.nanmean(cloud_frac[:, x0:x1, y0:y1], axis=(1, 2))
    else:
        local_clouds = cloud_percent.copy()
    
    logger.debug(f"Cloud stats - mean global: {np.mean(cloud_percent):.2f}, mean local: {np.mean(local_clouds):.2f}")
    
    # Filter very cloudy scenes (>50%)
    keep = cloud_percent <= CLOUD_HARD_DROP
    
    # Apply weighted threshold for moderate clouds
    weighted = cloud_percent.copy()
    moderate = cloud_percent > 0.30
    weighted[moderate] = 0.25 * cloud_percent[moderate] + 0.75 * local_clouds[moderate]
    keep &= (weighted <= CLOUD_FINAL_MAX)
    
    # Extract dates for kept scenes
    dates_ymd = np.array([int(np.datetime_as_string(t, unit="D").replace("-", ""))
                          for t in times], np.int64)
    
    # Month-level filtering
    kept_indices = np.where(keep)[0]
    final_indices = []
    
    for idx in kept_indices:
        date = dates_ymd[idx]
        month = date // 100
        
        # Check for better alternatives in same month
        same_month = (dates_ymd // 100) == month
        if np.sum(same_month) > 1:
            month_local = local_clouds[same_month]
            if np.min(month_local) < local_clouds[idx] - 0.20:
                continue  # Skip this one, better alternative exists
        
        final_indices.append(idx)
    
    keep_idx = np.array(final_indices, dtype=np.int64)
    
    logger.info(f"Cloud filtering: {len(items)} → {len(keep_idx)} clean scenes")
    
    # Extract clean steps (julian days matching legacy format)
    if len(keep_idx) > 0:
        clean_dates = times[keep_idx]
        clean_steps = np.array(extract_dates_legacy(clean_dates, year), dtype=np.int64)
    else:
        clean_steps = np.array([], dtype=np.int64)
    
    # Prepare legacy cloud output (scaled to 255, with NaN for >100)
    clouds_legacy = cloud_frac * 255.0
    clouds_legacy = np.where(clouds_legacy > 100, np.nan, clouds_legacy).astype(np.float32)
    
    return clouds_legacy, cloud_percent[keep_idx], clean_steps, local_clouds[keep_idx]

# ----------------------------
# Stage 2: Download Sentinel-2 (replaces download_sentinel_2_new)
# ----------------------------
def download_sentinel2_stac(items: List, bbox: list, clean_steps: np.ndarray,
                           year: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 2: Download Sentinel-2 data for clean dates.
    Replaces legacy download_sentinel_2_new function.
    
    Returns: (img_10, img_20, dates_array, cirrus_mask)
    """
    if len(items) == 0 or len(clean_steps) == 0:
        return (np.zeros((0,0,0,4), np.uint16), 
                np.zeros((0,0,0,6), np.uint16),
                np.array([], np.int64),
                np.zeros((0,0,0), np.float32))
    
    # Extract item dates and convert to julian days
    item_dates = []
    for item in items:
        dt = np.datetime64(item.datetime, 'D')
        item_dates.append(dt)
    
    item_julian = np.array(extract_dates_legacy(item_dates, year))
    
    # Match clean_steps to available items (within 3 days)
    steps_to_download = []
    dates_to_download = []
    filtered_items = []
    
    for step in clean_steps:
        closest_idx = np.argmin(np.abs(item_julian - step))
        time_diff = np.min(np.abs(item_julian - step))
        
        if time_diff < 3:
            steps_to_download.append(closest_idx)
            dates_to_download.append(item_julian[closest_idx])
            filtered_items.append(items[closest_idx])
        else:
            logger.warning(f"Date/orbit mismatch for step {step}, closest is {time_diff} days away")
    
    if len(filtered_items) == 0:
        logger.warning("No items matched clean_steps")
        return (np.zeros((0,0,0,4), np.uint16),
                np.zeros((0,0,0,6), np.uint16),
                np.array([], np.int64),
                np.zeros((0,0,0), np.float32))
    
    logger.info(f"Downloading {len(filtered_items)} scenes matching clean steps")
    
    # Load SCL for quality filtering (at 160m like legacy DATA_QUALITY)
    t_scl = time.perf_counter()
    ds_scl = stac_load(
        filtered_items,
        bands=SCL_ASSET,
        bbox=bbox,
        resolution=160,  # Match legacy quality resolution
        groupby="solar_day",
        resampling="nearest",
        dtype="uint8",
        chunks={}
    )
    logger.debug(f"Loaded SCL for quality in {_elapsed_ms(t_scl):.0f} ms")
    
    scl_quality = _to_numpy(ds_scl["scl"])  # (T,h,w)
    
    # Calculate quality metric (matching legacy logic)
    bad_pixels = np.zeros_like(scl_quality, dtype=bool)
    for code in SCL_CLOUDY:
        bad_pixels |= (scl_quality == code)
    
    quality_per_img = np.mean(bad_pixels, axis=(1, 2))
    
    # Extract cirrus before filtering (SCL value 10)
    cirrus_img = (scl_quality == 10).astype(np.float32)
    cirrus_img = remove_noise_clouds(cirrus_img)
    
    # Remove low quality images
    steps_to_rm = np.argwhere(quality_per_img > 0.2).flatten()
    
    if len(steps_to_rm) > 0:
        logger.info(f"Removing {len(steps_to_rm)} low quality images")
        keep_mask = np.ones(len(filtered_items), dtype=bool)
        keep_mask[steps_to_rm] = False
        filtered_items = [item for i, item in enumerate(filtered_items) if keep_mask[i]]
        dates_to_download = [d for i, d in enumerate(dates_to_download) if keep_mask[i]]
        cirrus_img = cirrus_img[keep_mask]
    
    if len(filtered_items) == 0:
        logger.warning("All images removed by quality filter")
        return (np.zeros((0,0,0,4), np.uint16),
                np.zeros((0,0,0,6), np.uint16),
                np.array([], np.int64),
                np.zeros((0,0,0), np.float32))
    
    # Load 10m bands
    t_10m = time.perf_counter()
    ds_10m = stac_load(
        filtered_items,
        bands=ASSETS_10M,
        bbox=bbox,
        resolution=10,
        groupby="solar_day",
        resampling="nearest",
        dtype="uint16",
        chunks={}
    )
    logger.info(f"Loaded 10m bands in {_elapsed_ms(t_10m):.0f} ms, shape: {ds_10m.sizes}")
    
    # Load 20m bands
    t_20m = time.perf_counter()
    ds_20m = stac_load(
        filtered_items,
        bands=ASSETS_20M,
        bbox=bbox,
        resolution=20,
        groupby="solar_day",
        resampling="nearest",
        dtype="uint16",
        chunks={}
    )
    logger.info(f"Loaded 20m bands in {_elapsed_ms(t_20m):.0f} ms, shape: {ds_20m.sizes}")
    
    # Process 10m bands
    img_10_list = []
    for band in ASSETS_10M:
        if band in ds_10m:
            data = _to_numpy(ds_10m[band])
            # Convert to float [0,1] then to uint16
            data_float = np.clip(data.astype(np.float32) / 10000.0, 0, 1)
            img_10_list.append(data_float)
    
    if img_10_list:
        img_10 = np.stack(img_10_list, axis=-1)
        img_10 = to_uint16(img_10)
    else:
        img_10 = np.zeros((len(filtered_items), 10, 10, 4), np.uint16)
    
    # Process 20m bands (only 6 bands, no 40m bands)
    img_20_list = []
    for band in ASSETS_20M:
        if band in ds_20m:
            data = _to_numpy(ds_20m[band])
            data_float = np.clip(data.astype(np.float32) / 10000.0, 0, 1)
            img_20_list.append(data_float)
    
    if img_20_list:
        img_20 = np.stack(img_20_list, axis=-1)
        img_20 = to_uint16(img_20)
    else:
        img_20 = np.zeros((len(filtered_items), 10, 10, 6), np.uint16)
    
    # # Resize cirrus to match output resolution
    # if cirrus_img.shape[1:] != img_20.shape[1:3]:
    #     cirrus_resized = np.zeros((cirrus_img.shape[0], img_20.shape[1], img_20.shape[2]))
    #     for i in range(cirrus_img.shape[0]):
    #         # Use repeat to match legacy upsampling
    #         scale_y = img_20.shape[1] // cirrus_img.shape[1]
    #         scale_x = img_20.shape[2] // cirrus_img.shape[2]
    #         if scale_y > 0 and scale_x > 0:
    #             cirrus_resized[i] = np.repeat(np.repeat(cirrus_img[i], scale_y, axis=0), scale_x, axis=1)
    #         else:
    #             # Downsample if needed
    #             cirrus_resized[i] = cirrus_img[i][:img_20.shape[1], :img_20.shape[2]]
    #     cirrus_img = cirrus_resized
    
    # Prepare dates array (as julian days)
    dates_array = np.array(dates_to_download, np.int64)
    
    logger.info(f"Final output shapes - 10m: {img_10.shape}, 20m: {img_20.shape}")
    
    return img_10, img_20, dates_array, cirrus_img.astype(np.float32)

# ----------------------------
# Main CLI
# ----------------------------
@click.command()
@click.option('--year', type=int, required=True, help='Year to process')
@click.option('--lon', type=float, required=True, help='Longitude of tile center')
@click.option('--lat', type=float, required=True, help='Latitude of tile center')
@click.option('--X_tile', 'x_tile', type=int, required=True, help='X tile index')
@click.option('--Y_tile', 'y_tile', type=int, required=True, help='Y tile index')
@click.option('--dest', type=str, required=True, help='Destination: local dir or s3://bucket/prefix')
@click.option('--expansion', type=int, default=300, help='Legacy expansion baseline (default=300)')
@click.option('--max-items', type=int, default=400, help='Maximum STAC items to search')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def main(year: int, lon: float, lat: float, x_tile: int, y_tile: int,
         dest: str, expansion: int, max_items: int, debug: bool):
    """
    Sentinel-2 L2A downloader with legacy-compatible outputs.
    Implements two-stage workflow: cloud identification → data download.
    """
    # Configure logging
    logger.remove()
    log_level = "DEBUG" if debug else "INFO"
    logger.add(sys.stderr, level=log_level, 
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    logger.info(f"Starting S2 download for tile {x_tile}X{y_tile}Y, year {year}")
    logger.info(f"Location: lon={lon}, lat={lat}")
    
    t_all = time.perf_counter()
    
    # Initialize storage
    if dest.startswith("s3://"):
        store = from_url(dest)
        logger.info(f"Using S3 storage: {dest}")
    else:
        os.makedirs(dest, exist_ok=True)
        store = LocalStore(prefix=dest)
        logger.info(f"Using local storage: {dest}")
    
    # Configure STAC/rio
    boto3_session = boto3.Session(region_name=S2_REGION)
    # Pass boto3 session (NOT rasterio.AWSSession) to odc/rasterio
    configure_rio(cloud_defaults=True, aws={"session": boto3_session})
    
    # Define bounding boxes (matching legacy logic)
    initial_bbx = [lon, lat, lon, lat]
    cloud_bbx = make_bbox(initial_bbx, expansion=(expansion * 15) // 30)  # Large area for clouds
    tile_bbx = make_bbox(initial_bbx, expansion=expansion // 30)  # Small area for data
    
    cloud_size_km = (cloud_bbx[2] - cloud_bbx[0]) * 111.32
    tile_size_km = (tile_bbx[2] - tile_bbx[0]) * 111.32
    
    logger.info(f"Cloud bbox: ~{cloud_size_km:.1f} km")
    logger.info(f"Tile bbox: ~{tile_size_km:.1f} km")
    
    # Setup paths
    paths = SavePaths(root=dest.rstrip("/"), year=year, X_tile=x_tile, Y_tile=y_tile)
    ensure_dirs(store, paths.clouds_dir, paths.misc_dir, paths.s2_10_dir, paths.s2_20_dir)
    
    # STAC client
    client = Client.open(EARTH_SEARCH_V1)
    date_range = f"{year}"

    # Helper: log and retry STAC searches with light backoff
    def _log_search_context(label: str, params: dict, search) -> None:
        try:
            url = search.url_with_parameters()
            logger.warning(f"{label} STAC URL: {url}")
        except Exception:
            logger.warning(f"{label} STAC params: {params}")

    def _search_items(
        *,
        label: str,
        collection: str,
        dt: str,
        geometry_kind: str,
        geometry_value,
        query: dict,
        limit: int,
        attempts: int = 5,
        initial_backoff: float = 0.5,
    ) -> List:
        params = {
            "collections": [collection],
            "datetime": dt,
            geometry_kind: geometry_value,
            "query": query,
            "limit": limit,
            "sortby":"+properties.eo:cloud_cover",
        }
        # Build search lazily each attempt to avoid internal state issues
        for attempt in range(1, attempts + 1):
            search = client.search(**params)
            if attempt == 1:
                _log_search_context(label, params, search)
            try:
                # small jitter to avoid thundering herd
                time.sleep(random.uniform(0.005, 0.2))
                items = search.item_collection()
                items = sorted(items, key=lambda it: it.datetime)
                return items
            except Exception as e:
                if attempt == attempts:
                    logger.error(f"{label} STAC failed after {attempts} attempts: {e}")
                    raise
                sleep_s = initial_backoff * (2 ** (attempt - 1))
                logger.warning(
                    f"{label} STAC error on attempt {attempt}/{attempts}: {e}; retrying in {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)
    
    # ========== STAGE 1: Cloud Identification ==========
    logger.info("=" * 50)
    logger.info("STAGE 1: Cloud identification (large area)")
    
    t_search = time.perf_counter()
    items_cloud = _search_items(
        label="CloudSearch",
        collection=S2_COLLECTION,
        dt=date_range,
        geometry_kind="intersects",
        geometry_value=bbox2geojson(cloud_bbx),
        query={"eo:cloud_cover": {"lt": 50}},
        limit=max_items,
    )
    logger.info(f"Found {len(items_cloud)} scenes in {_elapsed_ms(t_search):.0f} ms")
    
    if len(items_cloud) == 0:
        logger.warning("No scenes found for cloud analysis")
        # Write empty outputs
        obstore_put_hkl(store, paths.f_clean(), np.array([], np.int64))
        obstore_put_hkl(store, paths.f_clouds(), np.zeros((0,0,0), np.float32))
        obstore_put_hkl(store, paths.f_cloudmask(), np.zeros((0,0,0), np.uint8))
        obstore_put_hkl(store, paths.f_s2_dates(), np.array([], np.int64))
        obstore_put_hkl(store, paths.f_s2_10(), np.zeros((0,0,0,4), np.uint16))
        obstore_put_hkl(store, paths.f_s2_20(), np.zeros((0,0,0,6), np.uint16))
        return
    
    # Identify clean dates
    t_cloud = time.perf_counter()
    clouds_legacy, cloud_percent, clean_steps, local_clouds = identify_clouds_stac(
        items_cloud, cloud_bbx, year
    )
    logger.info(f"Cloud analysis completed in {_elapsed_ms(t_cloud):.0f} ms")
    logger.info(f"Identified {len(clean_steps)} clean dates")
    
    if len(clean_steps) > 0 and debug:
        logger.debug(f"Clean steps: {clean_steps[:min(5, len(clean_steps))]}")
        logger.debug(f"Cloud %: min={cloud_percent.min():.1f}, mean={cloud_percent.mean():.1f}")
    
    # Save cloud outputs
    obstore_put_hkl(store, paths.f_clouds(), clouds_legacy)
    obstore_put_hkl(store, paths.f_clean(), clean_steps)
    
    # ========== STAGE 2: Download Sentinel-2 Data ==========
    logger.info("=" * 50)
    logger.info("STAGE 2: Download Sentinel-2 data (tile area)")
    
    if len(clean_steps) == 0:
        logger.warning("No clean dates to download")
        obstore_put_hkl(store, paths.f_cloudmask(), np.zeros((0,0,0), np.uint8))
        obstore_put_hkl(store, paths.f_s2_dates(), np.array([], np.int64))
        obstore_put_hkl(store, paths.f_s2_10(), np.zeros((0,0,0,4), np.uint16))
        obstore_put_hkl(store, paths.f_s2_20(), np.zeros((0,0,0,6), np.uint16))
        return
    
    # Search tile area
    t_search = time.perf_counter()
    items_tile = _search_items(
        label="TileSearch",
        collection=S2_COLLECTION,
        dt=date_range,
        geometry_kind="intersects",
        geometry_value=bbox2geojson(tile_bbx),
        query={"eo:cloud_cover": {"lt": 100}},
        limit=max_items,
    )
    logger.info(f"Found {len(items_tile)} tile scenes in {_elapsed_ms(t_search):.0f} ms")
    
    # Download data for clean dates
    t_download = time.perf_counter()
    img_10, img_20, dates_array, cirrus_mask = download_sentinel2_stac(
        items_tile, tile_bbx, clean_steps, year
    )
    logger.info(f"Download completed in {_elapsed_ms(t_download):.0f} ms")
    
    # Convert dates to DOY for s2_dates file
    if len(dates_array) > 0:
        # Convert julian days back to dates then to DOY
        dates_for_doy = []
        for julian in dates_array:
            # Rough conversion back to date
            day_in_year = julian % 365
            if day_in_year > 0:
                dates_for_doy.append(day_in_year)
        s2_dates_doy = np.array(dates_for_doy, np.int64)
    else:
        s2_dates_doy = np.array([], np.int64)
    
    # Create cloudmask at 20m resolution from cirrus
    cloudmask_20 = (cirrus_mask > 0).astype(np.uint8)
    
    # Save outputs
    logger.info("Saving outputs...")
    t_save = time.perf_counter()
    
    obstore_put_hkl(store, paths.f_cloudmask(), cloudmask_20)
    obstore_put_hkl(store, paths.f_s2_dates(), s2_dates_doy)
    obstore_put_hkl(store, paths.f_s2_10(), img_10)
    obstore_put_hkl(store, paths.f_s2_20(), img_20)
    
    logger.info(f"Outputs saved in {_elapsed_ms(t_save):.0f} ms")
    
    # Summary
    total_time = _elapsed_ms(t_all)
    logger.info("=" * 50)
    logger.info(f"Processing complete in {total_time/1000:.1f} seconds")
    logger.info(f"Output location: {paths.base}")
    logger.info(f"Results: {len(clean_steps)} clean dates → {img_10.shape[0]} final images")
    logger.info(f"Data shapes - 10m: {img_10.shape}, 20m: {img_20.shape}")

# --- programmatic entrypoint for Lithops ---
def run(
    year: int | str,
    lon: float,
    lat: float,
    X_tile: int | str,
    Y_tile: int | str,
    dest: str,
    expansion: int = 300,
    max_items: int = 100,
    debug: bool = False,
) -> dict:
    """
    Programmatic wrapper for the Click command.
    We call the command's underlying callback directly to avoid sys.exit().
    """
    # NOTE: in Click, the decorated function is replaced by a Command object.
    # Its original Python function is available as `main.callback`.
    # The callback signature is: (year, lon, lat, x_tile, y_tile, dest, expansion, max_items, debug)
    main.callback(
        year=int(year),
        lon=float(lon),
        lat=float(lat),
        x_tile=int(X_tile),
        y_tile=int(Y_tile),
        dest=dest,
        expansion=int(expansion),
        max_items=int(max_items),
        debug=bool(debug),
    )

    return {
        "product": "s2",
        "year": int(year),
        "lon": float(lon),
        "lat": float(lat),
        "X_tile": int(X_tile),
        "Y_tile": int(Y_tile),
        "dest": dest,
        "expansion": int(expansion),
        "max_items": int(max_items),
    }


if __name__ == "__main__":
    main()