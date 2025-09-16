#!/usr/bin/env python3
"""
Load Sentinel-1 data from the STAC endpoint with terrain correction and gamma nought calibration.

Enhanced version with:
- Terrain-corrected gamma nought (γ0_terrain) calibration
- Quarterly composites instead of monthly
- DEM caching to avoid redundant downloads
- Calibration using ESA's XML metadata

/{YEAR}/raw/{X_tile}/{Y_tile}/raw/misc/s1_dates_{X_tile}X{Y_tile}Y.hkl
/{YEAR}/raw/{X_tile}/{Y_tile}/raw/s1/{X_tile}X{Y_tile}Y.hkl

# Args: --year --lon --lat --X_tile --Y_tile --dest [--expansion 300]
# Produces:
#   raw/s1/{X}X{Y}Y.hkl           (uint16, (12,H,W,2))  # 4 quarters x 3 repeats = 12
#   raw/misc/s1_dates_{tile}.hkl  (int, (12,), quarter dates)
"""
import os, sys, argparse, tempfile
from rasterio.session import AWSSession
import boto3
import rasterio as rio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as xf_from_bounds
from rasterio.features import bounds as featureBounds
from rasterio.warp import reproject
from rasterio.merge import merge
import xarray as xr
import numpy as np
from odc.stac import configure_rio
from pystac_client import Client
from shapely.geometry import shape, box
from datetime import datetime
from collections import defaultdict
import hickle as hkl
import time
from loguru import logger
import obstore as obs
from obstore.store import S3Store, LocalStore, from_url
import xml.etree.ElementTree as ET
import requests
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RectBivariateSpline
import hashlib
import pickle
from pathlib import Path
import json
from urllib.parse import urlparse

def _elapsed_ms(t_start: float) -> float:
    return (time.perf_counter() - t_start) * 1000.0

EARTH_SEARCH_V1 = "https://earth-search.aws.element84.com/v1"
S1_COLLECTION = "sentinel-1-grd"
S1_REGION = "eu-central-1"
COPDEM_COLLECTION = "cop-dem-glo-30"
BUCKET_REGION = "us-west-2"

ASSETS_S1 = ["vv", "vh"]

# DEM cache directory (robust for read-only HOME on Lambda)
def _resolve_cache_dir() -> Path | None:
    candidates = [
        os.environ.get("XDG_CACHE_HOME"),
        os.environ.get("CACHE_DIR"),
        "/tmp/.cache",
    ]
    for root in candidates:
        if not root:
            continue
        try:
            p = Path(root) / "sentinel1_dem"
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            continue
    return None

DEM_CACHE_DIR = _resolve_cache_dir()

# bbox helper (expand a point bbox by ~degrees)
def make_bbox(initial_bbx: list, expansion: int = 10) -> list:
    multiplier = 1/360
    bbx = initial_bbx.copy()
    bbx[0] -= expansion * multiplier
    bbx[1] -= expansion * multiplier
    bbx[2] += expansion * multiplier
    bbx[3] += expansion * multiplier
    return bbx

def bbox2geojson(bbox: list) -> dict:
    x1, y1, x2, y2 = bbox
    return {
        "type": "Polygon",
        "coordinates": [[[x1,y1],[x2,y1],[x2,y2],[x1,y2],[x1,y1]]]
    }

def obstore_put_hkl(store, relpath: str, obj) -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".hkl", delete=False)
    tmp.close()
    try:
        hkl.dump(obj, tmp.name, mode="w", compression="gzip")
        with open(tmp.name, "rb") as f:
            obs.put(store, relpath, f.read())
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass

def ensure_dirs(store, *dirs: str) -> None:
    for d in dirs:
        try:
            obs.put(store, d.rstrip("/") + "/.keep", b"")
        except Exception:
            pass

class DEMCache:
    """Caches DEM data to avoid redundant downloads"""
    
    @staticmethod
    def get_cache_key(bbox: list) -> str:
        """Generate a unique cache key for a bbox"""
        bbox_str = f"{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}"
        return hashlib.md5(bbox_str.encode()).hexdigest()
    
    @staticmethod
    def get_cached_dem(bbox: list) -> tuple:
        """Try to load cached DEM data"""
        if DEM_CACHE_DIR is not None:
            cache_key = DEMCache.get_cache_key(bbox)
            cache_file = DEM_CACHE_DIR / f"dem_{cache_key}.pkl"
            if cache_file.exists():
                logger.info(f"Loading cached DEM from {cache_file}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        return None
    
    @staticmethod
    def save_dem_cache(bbox: list, dem_data: np.ndarray, dem_transform, dem_crs,
                       slope: np.ndarray, aspect: np.ndarray) -> None:
        """Save DEM data to cache"""
        if DEM_CACHE_DIR is None:
            logger.debug("DEM cache dir unavailable; skipping cache save")
            return None
        try:
            cache_key = DEMCache.get_cache_key(bbox)
            cache_file = DEM_CACHE_DIR / f"dem_{cache_key}.pkl"
            logger.info(f"Caching DEM to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump((dem_data, dem_transform, dem_crs, slope, aspect), f)
        except Exception as e:
            logger.warning(f"Failed to write DEM cache: {e}")

def fetch_copdem_for_bbox(bbox: list, buffer: float = 0.01) -> tuple:
    """Fetch COP-DEM-GLO-30 data for the given bounding box with caching"""
    
    # Check cache first
    cached = DEMCache.get_cached_dem(bbox)
    if cached is not None:
        return cached
    
    logger.info(f"Fetching COP-DEM for bbox: {bbox}")
    
    # Add buffer
    buffered_bbox = [
        bbox[0] - buffer,
        bbox[1] - buffer,
        bbox[2] + buffer,
        bbox[3] + buffer
    ]
    
    client = Client.open(EARTH_SEARCH_V1)
    
    # Search for DEM tiles
    search = client.search(
        collections=[COPDEM_COLLECTION],
        bbox=buffered_bbox
    )
    
    items = list(search.items())
    
    if not items:
        logger.warning("No COP-DEM data found for the specified area")
        return None, None, None, None, None
    
    logger.info(f"Found {len(items)} DEM tiles")
    
    if len(items) == 1:
        # Single tile
        with rio.open(items[0].assets["data"].href) as src:
            dem_data = src.read(1)
            dem_transform = src.transform
            dem_crs = src.crs
    else:
        # Multiple tiles - mosaic
        logger.info("Mosaicking multiple DEM tiles")
        src_files = []
        for item in items:
            src_files.append(rio.open(item.assets["data"].href))
        
        mosaic, out_transform = merge(src_files, bounds=buffered_bbox)
        dem_data = mosaic[0]
        dem_transform = out_transform
        dem_crs = 'EPSG:4326'
        
        for src in src_files:
            src.close()
    
    # Calculate slope and aspect
    slope, aspect = calculate_slope_aspect(dem_data, dem_transform)
    
    # Cache the results
    DEMCache.save_dem_cache(bbox, dem_data, dem_transform, dem_crs, slope, aspect)
    
    return dem_data, dem_transform, dem_crs, slope, aspect

def calculate_slope_aspect(dem: np.ndarray, transform, resolution: float = 30) -> tuple:
    """Calculate slope and aspect from DEM"""
    logger.debug("Calculating slope and aspect from DEM")
    
    # Calculate gradients
    dy, dx = np.gradient(dem, resolution)
    
    # Calculate slope (in radians)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    
    # Calculate aspect (in radians, 0 = North, clockwise)
    aspect = np.arctan2(-dx, dy)
    aspect = np.where(aspect < 0, aspect + 2*np.pi, aspect)
    
    logger.debug(f"Slope range: {np.degrees(slope.min()):.1f} to {np.degrees(slope.max()):.1f} degrees")
    
    return slope, aspect

def s3_to_https(s3_url: str, region: str = S1_REGION) -> str:
    """
    Convert an S3 URI (s3://bucket/key) to a public HTTPS URL 
    for accessing via requests library.
    """
    parsed = urlparse(s3_url)
    
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")  # remove leading slash
        return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    elif parsed.scheme in ["http", "https"]:
        # Already an HTTP URL
        return s3_url
    else:
        logger.warning(f"Unknown URL scheme: {parsed.scheme} in {s3_url}")
        return s3_url

def parse_calibration_xml(calibration_url: str, calibration_type: str = 'gamma') -> dict:
    """Parse Sentinel-1 calibration XML to extract LUT"""
    # Convert S3 URI to HTTPS if needed
    https_url = s3_to_https(calibration_url)
    logger.debug(f"Fetching calibration XML from {https_url}")
    
    try:
        response = requests.get(https_url, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        
        logger.debug(f"XML root tag: {root.tag}")
        
        # Try different paths for calibration data
        calib_list = root.find('.//calibrationVectorList')
        if calib_list is None:
            # Try alternative paths
            for path in ['.//calibrationVectorList', 'calibrationVectorList', 
                        './/calibration/calibrationVectorList']:
                calib_list = root.find(path)
                if calib_list is not None:
                    logger.debug(f"Found calibration data at path: {path}")
                    break
        
        if calib_list is None:
            logger.warning("No calibration data found in XML")
            # Log XML structure for debugging
            logger.debug(f"XML structure: {ET.tostring(root, encoding='unicode')[:500]}")
            return None
        
        lut_data = {
            'lines': [],
            'pixels': [],
            'values': []
        }
        
        vectors = calib_list.findall('calibrationVector')
        logger.debug(f"Found {len(vectors)} calibration vectors")
        
        for vector in vectors:
            line = int(vector.find('line').text)
            pixel_list = [int(p) for p in vector.find('pixel').text.split()]
            
            # Find the correct calibration values
            cal_element = vector.find(calibration_type)
            if cal_element is None:
                cal_element = vector.find(f'{calibration_type}Nought')
            if cal_element is None and calibration_type == 'gamma':
                cal_element = vector.find('gamma0')
            
            if cal_element is not None:
                value_list = [float(v) for v in cal_element.text.split()]
            else:
                logger.warning(f"No {calibration_type} values found in vector")
                continue
            
            lut_data['lines'].append(line)
            lut_data['pixels'].append(pixel_list)
            lut_data['values'].append(value_list)
        
        if not lut_data['lines']:
            logger.warning("No valid calibration vectors found")
            return None
            
        logger.debug(f"Parsed {len(lut_data['lines'])} calibration lines")
        return lut_data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch calibration XML: {e}")
        logger.debug(f"Original URL: {calibration_url}")
        logger.debug(f"HTTPS URL: {https_url}")
        return None
    except Exception as e:
        logger.error(f"Failed to parse calibration XML: {e}")
        return None

def interpolate_lut_to_image(lut_data: dict, image_shape: tuple) -> np.ndarray:
    """Interpolate calibration LUT to full image size"""
    h, w = image_shape
    calibration_grid = np.zeros((h, w), dtype=np.float32)
    
    if lut_data is None:
        logger.warning("No LUT data, using unit calibration")
        calibration_grid.fill(1.0)
        return calibration_grid
    
    logger.debug(f"Interpolating LUT to image shape {image_shape}")
    
    # Create sparse grid
    points_added = 0
    for i, line in enumerate(lut_data['lines']):
        if line < h:
            for pixel, value in zip(lut_data['pixels'][i], lut_data['values'][i]):
                if pixel < w:
                    calibration_grid[line, pixel] = value
                    points_added += 1
    
    logger.debug(f"Added {points_added} calibration points to grid")
    
    # Fill empty values with nearest neighbor interpolation
    mask = calibration_grid > 0
    if np.any(mask):
        indices = distance_transform_edt(~mask, return_indices=True)[1]
        calibration_grid = calibration_grid[tuple(indices)]
        logger.debug(f"Calibration grid range: {calibration_grid.min():.2f} to {calibration_grid.max():.2f}")
    else:
        logger.warning("No valid calibration points, using unit calibration")
        calibration_grid.fill(1.0)
    
    return calibration_grid

def apply_radiometric_calibration(dn_data: np.ndarray, calibration_lut: np.ndarray) -> np.ndarray:
    """Apply radiometric calibration to convert DN to gamma0"""
    logger.debug(f"Applying calibration - DN range: {dn_data.min():.1f} to {dn_data.max():.1f}")
    
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma0 = (dn_data.astype(np.float32) ** 2) / (calibration_lut ** 2)
        gamma0 = np.where(calibration_lut > 0, gamma0, 0)
    
    valid_mask = gamma0 > 0
    if np.any(valid_mask):
        logger.debug(f"Gamma0 range: {gamma0[valid_mask].min():.2e} to {gamma0[valid_mask].max():.2e}")
    else:
        logger.warning("No valid gamma0 values after calibration")
    
    return gamma0

def apply_terrain_flattening(gamma0: np.ndarray, slope: np.ndarray, aspect: np.ndarray,
                            incidence_angle: float, look_direction: float) -> np.ndarray:
    """Apply terrain flattening to convert gamma0 to gamma0_terrain"""
    logger.debug(f"Applying terrain flattening (incidence: {incidence_angle:.1f}°)")
    
    # Convert to radians
    theta_i = np.radians(incidence_angle)
    
    # Calculate local incidence angle
    cos_local_inc = (np.cos(slope) * np.cos(theta_i) + 
                    np.sin(slope) * np.sin(theta_i) * 
                    np.cos(aspect - look_direction))
    
    # Avoid division by zero
    cos_local_inc = np.where(cos_local_inc > 0.1, cos_local_inc, 0.1)
    
    # Reference incidence angle
    cos_ref_inc = np.cos(theta_i)
    
    # Apply terrain flattening
    gamma0_terrain = gamma0 * (cos_ref_inc / cos_local_inc)
    
    # Mask invalid corrections
    invalid_mask = (cos_local_inc < 0.2) | (gamma0_terrain < 0)
    gamma0_terrain = np.where(invalid_mask, 0, gamma0_terrain)
    
    valid_ratio = np.sum(~invalid_mask) / invalid_mask.size
    logger.debug(f"Valid pixels after terrain correction: {valid_ratio:.1%}")
    
    return gamma0_terrain

def get_item_geometry(item) -> tuple:
    """Extract incidence angle and look direction from STAC item"""
    incidence_angle = item.properties.get('sar:incidence_angle',
                                         item.properties.get('view:incidence_angle', 35))
    
    orbit_state = item.properties.get('sat:orbit_state', '').upper()
    if orbit_state == 'ASCENDING':
        look_direction = np.pi / 2  # East (90°)
    else:  # DESCENDING
        look_direction = 3 * np.pi / 2  # West (270°)
    
    logger.debug(f"Scene geometry - Incidence: {incidence_angle}°, Orbit: {orbit_state}")
    
    return float(incidence_angle), look_direction

def find_calibration_asset(item, band: str) -> str:
    """Find the correct calibration asset name for a band"""
    # List all available assets for debugging
    logger.debug(f"Available assets for {item.id}: {list(item.assets.keys())}")
    
    # Try different naming conventions
    possible_names = [
        f"{band}-calibration",
        f"calibration-{band}",
        f"{band}_calibration",
        f"calibration_{band}",
        "calibration",  # Sometimes it's a single asset for all bands
        f"{band}-cal",
        f"cal-{band}"
    ]
    
    for name in possible_names:
        if name in item.assets:
            logger.debug(f"Found calibration asset: {name}")
            return name
    
    # Check if there's a general calibration asset with band info in description
    for asset_name, asset in item.assets.items():
        if 'calibration' in asset_name.lower() or 'cal' in asset_name.lower():
            logger.debug(f"Found potential calibration asset: {asset_name}")
            return asset_name
    
    return None

def process_band_with_terrain_correction(
    data_href: str,
    calibration_href: str,
    bounds: tuple,
    target_crs: str,
    dem_data: np.ndarray,
    dem_transform,
    dem_crs,
    slope: np.ndarray,
    aspect: np.ndarray,
    incidence_angle: float,
    look_direction: float,
    aws_session
) -> np.ndarray:
    """Process a single band with calibration and terrain correction"""
    
    logger.debug(f"Processing band with terrain correction")
    logger.debug(f"Data URL: {data_href}")
    logger.debug(f"Calibration URL: {calibration_href}")
    
    with rio.Env(aws_session):
        # Read the raw data
        with rio.open(data_href) as src:
            logger.debug(f"Source CRS: {src.crs}, Shape: {src.shape}")
            
            with WarpedVRT(
                src,
                crs=target_crs,
                resampling=Resampling.nearest,
                dst_nodata=0,
            ) as vrt:
                win = from_bounds(*bounds, transform=vrt.transform)
                win = win.round_offsets().round_lengths()
                
                dn_data = vrt.read(1, window=win).astype(np.float32)
                sar_transform = vrt.window_transform(win)
                sar_shape = dn_data.shape
                
                logger.debug(f"Read data shape: {sar_shape}, DN range: {dn_data.min():.1f} to {dn_data.max():.1f}")
    
    # Parse and apply calibration
    if calibration_href:
        lut_data = parse_calibration_xml(calibration_href, 'gamma')
        if lut_data:
            calibration_lut = interpolate_lut_to_image(lut_data, sar_shape)
            gamma0 = apply_radiometric_calibration(dn_data, calibration_lut)
        else:
            logger.warning("Failed to parse calibration, using DN values directly")
            gamma0 = dn_data
    else:
        logger.warning("No calibration URL provided, using DN values directly")
        gamma0 = dn_data
    
    if dem_data is not None:
        logger.debug("Applying terrain correction")
        # Reproject DEM data to match SAR geometry
        dem_matched = np.zeros(sar_shape, dtype=np.float32)
        slope_matched = np.zeros(sar_shape, dtype=np.float32)
        aspect_matched = np.zeros(sar_shape, dtype=np.float32)
        
        for source, destination, name in [
            (dem_data, dem_matched, "DEM"),
            (slope, slope_matched, "slope"),
            (aspect, aspect_matched, "aspect")
        ]:
            reproject(
                source=source,
                destination=destination,
                src_transform=dem_transform,
                src_crs=dem_crs,
                dst_transform=sar_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )
            logger.debug(f"Reprojected {name} to SAR geometry")
        
        # Apply terrain flattening
        gamma0_terrain = apply_terrain_flattening(
            gamma0, slope_matched, aspect_matched,
            incidence_angle, look_direction
        )
    else:
        logger.debug("No DEM available, skipping terrain correction")
        gamma0_terrain = gamma0
    
    # Convert to uint16 for storage
    valid_mask = gamma0_terrain > 0
    if np.any(valid_mask):
        # Use percentiles to avoid outliers
        p1, p99 = np.percentile(gamma0_terrain[valid_mask], [0.1, 99.9])
        logger.debug(f"Scaling from [{p1:.2e}, {p99:.2e}] to uint16")
        p0, p25, p50, p75, p100 = np.percentile(gamma0_terrain[valid_mask], [0, 25, 50, 75, 100])
        logger.debug(f"Percentiles: {p0:.2e}, {p25:.2e}, {p50:.2e}, {p75:.2e}, {p100:.2e}")
        
        # Scale to uint16 range
        scaled = np.clip((gamma0_terrain - p1) / (p99 - p1 + 1e-10), 0, 1) * 65535
        #scaled = np.clip(gamma0_terrain, 0, 1) * 65535
    else:
        logger.warning("No valid data after processing, returning zeros")
        scaled = np.zeros_like(gamma0_terrain)
    
    result = scaled.astype(np.uint16)
    logger.debug(f"Final uint16 range: {result.min()} to {result.max()}")
    
    return result

def process_band_simple(
    data_href: str,
    bounds: tuple,
    target_crs: str,
    aws_session
) -> np.ndarray:
    """Process a band without calibration or terrain correction"""
    
    logger.debug(f"Processing band (simple mode)")
    logger.debug(f"Data URL: {data_href}")
    
    with rio.Env(aws_session):
        with rio.open(data_href) as src:
            logger.debug(f"Source shape: {src.shape}, CRS: {src.crs}")
            
            with WarpedVRT(
                src,
                crs=target_crs,
                resampling=Resampling.nearest,
                dst_nodata=0,
            ) as vrt:
                win = from_bounds(*bounds, transform=vrt.transform)
                win = win.round_offsets().round_lengths()
                
                data = vrt.read(1, window=win)
                logger.debug(f"Read data shape: {data.shape}, range: {data.min()} to {data.max()}")
                
                return data.astype(np.uint16)

def get_quarterly_scenes(items: list, year: int) -> dict:
    """Select one scene per quarter"""
    quarters = {
        'Q1': (f'{year}-01-15', f'{year}-03-15'),
        'Q2': (f'{year}-04-15', f'{year}-06-15'),
        'Q3': (f'{year}-07-15', f'{year}-09-15'),
        'Q4': (f'{year}-10-15', f'{year}-12-15')
    }
    
    quarterly_items = {}
    
    for q_name, (start, end) in quarters.items():
        start_dt = np.datetime64(start)
        end_dt = np.datetime64(end)
        
        # Find items in this quarter
        quarter_items = []
        for item in items:
            item_dt = np.datetime64(item.properties["datetime"])
            if start_dt <= item_dt <= end_dt:
                quarter_items.append(item)
        
        # Select the first item in the quarter (or modify selection strategy)
        if quarter_items:
            # Sort by date and take the first
            quarter_items.sort(key=lambda x: x.properties["datetime"])
            quarterly_items[q_name] = quarter_items[0]
            logger.info(f"{q_name}: Selected scene from {quarter_items[0].properties['datetime']}")
        else:
            logger.warning(f"{q_name}: No scenes found")
            quarterly_items[q_name] = None
    
    return quarterly_items

def main():
    ap = argparse.ArgumentParser(description="Load Sentinel-1 data with terrain correction (quarterly, obstore).")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--X_tile", type=int, required=True)
    ap.add_argument("--Y_tile", type=int, required=True)
    ap.add_argument("--dest", type=str, required=True)
    ap.add_argument("--expansion", type=int, default=300)
    ap.add_argument("--no-terrain-correction", action="store_true", help="Disable terrain correction")
    ap.add_argument("--no-calibration", action="store_true", help="Disable radiometric calibration")
    ap.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = ap.parse_args()

    logger.remove()
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    t_all = time.perf_counter()

    # AWS + rasterio config
    boto3_session = boto3.Session(region_name=S1_REGION)
    aws_session = AWSSession(boto3_session, requester_pays=True)
    configure_rio(cloud_defaults=True)

    # Setup output store (supports s3://bucket/prefix and local paths)
    if args.dest.startswith("s3://"):
        store = from_url(args.dest, region = "us-west-2")
    #if isinstance(store, LocalStore):
    else:
        os.makedirs(args.dest, exist_ok=True)
        #os.makedirs(store.prefix, exist_ok=True)
        store = LocalStore(prefix=args.dest)
    
    base_key = f"{args.year}/raw/{args.X_tile}/{args.Y_tile}/raw"
    s1_dir_key = f"{base_key}/s1"
    misc_key = f"{base_key}/misc"
    ensure_dirs(store, s1_dir_key, misc_key)
    fn_s1_key = f"{s1_dir_key}/{args.X_tile}X{args.Y_tile}Y.hkl"
    fn_s1_dates_key = f"{misc_key}/s1_dates_{args.X_tile}X{args.Y_tile}Y.hkl"

    # Query STAC
    initial_bbx = [args.lon, args.lat, args.lon, args.lat]
    bbx = make_bbox(initial_bbx, expansion=args.expansion/30)
    client = Client.open(EARTH_SEARCH_V1)

    search = client.search(
        collections=[S1_COLLECTION],
        datetime=f"{args.year}",
        intersects=bbox2geojson(bbx),
        limit=500,
    )

    items = list(search.items())
    if not items:
        raise RuntimeError("No Sentinel-1 items found for query.")

    logger.info(f"Found {len(items)} Sentinel-1 scenes for {args.year}")

    # Get DEM data if terrain correction is enabled
    dem_data = dem_transform = dem_crs = slope = aspect = None
    if not args.no_terrain_correction:
        dem_data, dem_transform, dem_crs, slope, aspect = fetch_copdem_for_bbox(bbx)
        if dem_data is None:
            logger.warning("DEM fetch failed, proceeding without terrain correction")

    # Select quarterly scenes
    quarterly_items = get_quarterly_scenes(items, args.year)
    
    # Determine which polarizations exist
    akeys = set()
    for item in quarterly_items.values():
        if item:
            akeys.update(item.assets.keys())
    
    # Debug: Show all available asset keys
    logger.debug(f"All available asset keys across quarters: {sorted(akeys)}")
    
    bands = [b for b in ASSETS_S1 if b in akeys]
    if not bands:
        logger.warning("Standard band names not found, trying to detect...")
        # Try to find VV/VH with different naming
        for key in akeys:
            if 'vv' in key.lower() and 'vv' not in bands:
                bands.append('vv')
            elif 'vh' in key.lower() and 'vh' not in bands:
                bands.append('vh')
    
    if not bands:
        bands = ["vv"]  # fallback
        logger.warning("Could not detect bands, using fallback: vv")

    logger.info(f"Processing bands: {bands}")

    target_crs = "EPSG:4326"
    bounds = featureBounds(bbox2geojson(bbx))

    # Process each quarter
    quarterly_arrays = []
    
    for q_name in ['Q1', 'Q2', 'Q3', 'Q4']:
        item = quarterly_items.get(q_name)
        
        if item is None:
            logger.warning(f"No data for {q_name}, using zeros")
            # Create dummy data matching expected shape
            if quarterly_arrays:
                h, w = quarterly_arrays[0].shape[1:3]
            else:
                h, w = 512, 512  # Default size
            quarter_data = np.zeros((len(bands), h, w), dtype=np.uint16)
        else:
            logger.info(f"Processing {q_name}: {item.id}")
            logger.debug(f"  Available assets: {list(item.assets.keys())}")
            
            # Get geometry for this scene
            incidence_angle, look_direction = get_item_geometry(item)
            
            # Process each band
            band_arrays = []
            for band in bands:
                logger.debug(f"  Looking for band: {band}")
                
                # Check if band exists in assets
                if band not in item.assets:
                    logger.warning(f"    Band {band} not found in assets")
                    # Try alternative naming
                    found = False
                    for asset_name in item.assets:
                        if band in asset_name.lower():
                            logger.info(f"    Using alternative asset name: {asset_name} for band {band}")
                            band = asset_name
                            found = True
                            break
                    
                    if not found:
                        logger.warning(f"    Could not find {band} in any form, using zeros")
                        if band_arrays:
                            shape = band_arrays[0].shape
                        else:
                            shape = (512, 512)
                        band_arrays.append(np.zeros(shape, dtype=np.uint16))
                        continue
                
                # Process the band
                if band in item.assets:
                    logger.debug(f"    Processing band {band}")
                    
                    # Try to find calibration asset - use the original band name (vv or vh)
                    cal_asset_name = None
                    if not args.no_calibration:
                        cal_asset_name = find_calibration_asset(item, band)
                    
                    if cal_asset_name and not args.no_calibration:
                        cal_href = item.assets[cal_asset_name].href
                        logger.debug(f"    Using calibration from: {cal_asset_name}")
                    else:
                        cal_href = None
                        if not args.no_calibration:
                            logger.warning(f"    No calibration found for {band}")
                    
                    if dem_data is not None and not args.no_terrain_correction and cal_href:
                        # Full processing with calibration and terrain correction
                        arr = process_band_with_terrain_correction(
                            item.assets[band].href,
                            cal_href,
                            bounds,
                            target_crs,
                            dem_data, dem_transform, dem_crs,
                            slope, aspect,
                            incidence_angle, look_direction,
                            aws_session
                        )
                    else:
                        # Simple processing
                        arr = process_band_simple(
                            item.assets[band].href,
                            bounds,
                            target_crs,
                            aws_session
                        )
                    
                    band_arrays.append(arr)
                else:
                    logger.warning(f"    Band {band} not available")
                    if band_arrays:
                        shape = band_arrays[0].shape
                    else:
                        shape = (512, 512)
                    band_arrays.append(np.zeros(shape, dtype=np.uint16))
            
            if band_arrays:
                quarter_data = np.stack(band_arrays, axis=0)
                logger.debug(f"  Quarter data shape: {quarter_data.shape}, range: {quarter_data.min()} to {quarter_data.max()}")
            else:
                logger.error(f"  No bands processed for {q_name}")
                quarter_data = np.zeros((len(bands), 512, 512), dtype=np.uint16)
        
        quarterly_arrays.append(quarter_data)
    
    # Ensure all quarterly arrays have the same shape by padding with reflection
    if quarterly_arrays:
        max_bands = max(arr.shape[0] for arr in quarterly_arrays)
        max_h = max(arr.shape[1] for arr in quarterly_arrays)
        max_w = max(arr.shape[2] for arr in quarterly_arrays)

        def _pad_quarter_array(arr, target_shape):
            pad_b = target_shape[0] - arr.shape[0]
            pad_h = target_shape[1] - arr.shape[1]
            pad_w = target_shape[2] - arr.shape[2]

            # Pad band dimension (if needed) using edge values
            if pad_b > 0:
                arr = np.pad(arr, ((0, pad_b), (0, 0), (0, 0)), mode='edge')

            # Pad spatial dimensions using reflect (fallback to edge if too small)
            if pad_h > 0 or pad_w > 0:
                spatial_mode = 'reflect' if (arr.shape[1] > 1 and arr.shape[2] > 1) else 'edge'
                arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode=spatial_mode)

            return arr

        target_shape = (max_bands, max_h, max_w)
        quarterly_arrays = [_pad_quarter_array(a, target_shape) for a in quarterly_arrays]
    
    # Stack quarters and repeat each 3 times to get 12 time steps
    # Shape: (4 quarters, bands, H, W)
    all_quarters = np.stack(quarterly_arrays, axis=0)
    logger.debug(f"All quarters shape: {all_quarters.shape}")
    
    # Repeat each quarter 3 times: Q1,Q1,Q1,Q2,Q2,Q2,Q3,Q3,Q3,Q4,Q4,Q4
    repeated = np.repeat(all_quarters, 3, axis=0)  # (12, bands, H, W)
    
    # Transpose to (12, H, W, bands) to match expected format
    final_array = np.transpose(repeated, (0, 2, 3, 1))
    
    # Write via obstore
    t_write = time.perf_counter()
    obstore_put_hkl(store, fn_s1_key, final_array)
    
    # Write quarterly dates (repeated 3 times each)
    # Using day 45, 135, 225, 315 to represent quarters
    quarter_days = [45, 135, 225, 315]
    s1_dates = np.repeat(quarter_days, 3).astype(np.int64)
    
    obstore_put_hkl(store, fn_s1_dates_key, s1_dates)
    
    full_s1_path = f"{args.dest.rstrip('/')}/{fn_s1_key}"
    full_dates_path = f"{args.dest.rstrip('/')}/{fn_s1_dates_key}"
    
    logger.success(f"S1 quarterly composites saved: {full_s1_path}")
    logger.success(f"Shape: {final_array.shape}, dtype: {final_array.dtype}")
    logger.success(f"Data range: {final_array.min()} to {final_array.max()}")
    logger.success(f"S1 quarterly dates saved: {full_dates_path}")
    logger.info(f"Write completed in {_elapsed_ms(t_write):.0f} ms; total runtime {_elapsed_ms(t_all):.0f} ms")

# --- programmatic entrypoint for Lithops ---
def run(
    year: int | str,
    lon: float,
    lat: float,
    X_tile: int | str,
    Y_tile: int | str,
    dest: str,
    expansion: int = 300,
    debug: bool = False,
    no_terrain_correction: bool = False,
    no_calibration: bool = False,
) -> dict:
    """
    Programmatic wrapper around the CLI for serverless execution.
    Synthesizes argv for argparse-based main().
    """
    import sys

    argv = [
        __file__,
        "--year", str(year),
        "--lon", str(lon),
        "--lat", str(lat),
        "--X_tile", str(X_tile),
        "--Y_tile", str(Y_tile),
        "--dest", dest,
        "--expansion", str(expansion),
    ]
    if no_terrain_correction:
        argv.append("--no-terrain-correction")
    if no_calibration:
        argv.append("--no-calibration")
    if debug:
        argv.append("--debug")

    old_argv = sys.argv
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old_argv

    return {
        "product": "s1",
        "year": int(year),
        "lon": float(lon),
        "lat": float(lat),
        "X_tile": int(X_tile),
        "Y_tile": int(Y_tile),
        "dest": dest,
        "no_terrain_correction": bool(no_terrain_correction),
        "no_calibration": bool(no_calibration),
    }


if __name__ == "__main__":
    main()