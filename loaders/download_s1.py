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
import os
import sys
import argparse
import tempfile
from rasterio.session import AWSSession
import boto3
import rasterio as rio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.features import bounds as featureBounds
from rasterio.warp import reproject, transform_bounds
from rasterio.merge import merge
from rasterio.crs import CRS
import numpy as np
from odc.stac import configure_rio
from pystac_client import Client
from shapely.geometry import shape, box
from datetime import datetime
import hickle as hkl
import time
from loguru import logger
import obstore as obs
from obstore.store import LocalStore, from_url
import xml.etree.ElementTree as ET
import requests
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
import hashlib
import pickle
from pathlib import Path
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

def coverage_fraction(item, tile_bounds: tuple) -> float:
    """Compute fraction of tile area covered by the item's footprint.

    Returns 0.0 on error.
    """
    try:
        tile_poly = box(*tile_bounds)
        item_poly = shape(item.geometry)
        inter = tile_poly.intersection(item_poly)
        if tile_poly.is_empty or tile_poly.area == 0:
            return 0.0
        return float(inter.area / tile_poly.area)
    except Exception as e:
        logger.warning(f"Failed computing coverage fraction for item {getattr(item, 'id', 'unknown')}: {e}")
        return 0.0

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

def obstore_put_text(store, relpath: str, text: str) -> None:
    try:
        obs.put(store, relpath, text.encode("utf-8"))
    except Exception as e:
        logger.error(f"Failed to write text sidecar {relpath}: {e}")

def compute_band_stats(arr: np.ndarray) -> dict:
    """Compute summary stats ignoring zeros as nodata.

    Returns dict with min, max, mean, std, p5, p50, p95, valid_ratio, count, valid_count.
    """
    vals = arr.astype(np.float32)
    total = int(vals.size)
    mask = vals > 0
    valid = vals[mask]
    if valid.size == 0:
        return {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "std": 0.0,
            "p5": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "valid_ratio": 0.0,
            "count": total,
            "valid_count": 0,
        }
    p5, p50, p95 = np.percentile(valid, [5, 50, 95])
    return {
        "min": int(valid.min()),
        "max": int(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
        "valid_ratio": float(valid.size / total) if total > 0 else 0.0,
        "count": total,
        "valid_count": int(valid.size),
    }

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

def fetch_copdem_for_bbox(bbox: list, buffer: float = 0.00002) -> tuple:
    """Fetch COP-DEM-GLO-30 data for the given bounding box with caching"""
    
    # Check cache first
    cached = DEMCache.get_cached_dem(bbox)
    if cached is not None:
        logger.info(f"Using cached DEM for bbox: {bbox}")
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
        # Single tile - windowed read over buffered_bbox
        with rio.open(items[0].assets["data"].href) as src:
            try:
                if src.crs and str(src.crs).upper() != 'EPSG:4326':
                    read_bounds = transform_bounds('EPSG:4326', src.crs, *buffered_bbox, densify_pts=21)
                else:
                    read_bounds = tuple(buffered_bbox)
            except Exception as e:
                logger.warning(f"DEM bounds transform failed ({e}); using original bbox")
                read_bounds = tuple(buffered_bbox)

            win = from_bounds(*read_bounds, transform=src.transform)
            win = win.round_offsets().round_lengths()

            dem_data = src.read(1, window=win)
            dem_transform = src.window_transform(win)
            dem_crs = src.crs
    else:
        # Multiple tiles - mosaic
        logger.info("Mosaicking multiple DEM tiles")
        src_files = []
        vrts = []
        try:
            for item in items:
                ds = rio.open(item.assets["data"].href)
                src_files.append(ds)
                if ds.crs and str(ds.crs).upper() != 'EPSG:4326':
                    vrts.append(WarpedVRT(ds, crs='EPSG:4326', resampling=Resampling.nearest))
                else:
                    vrts.append(ds)

            mosaic, out_transform = merge(vrts, bounds=buffered_bbox)
            dem_data = mosaic[0]
            dem_transform = out_transform
            dem_crs = 'EPSG:4326'
        finally:
            for v in vrts:
                try:
                    v.close()
                except Exception:
                    pass
            for src in src_files:
                try:
                    src.close()
                except Exception:
                    pass
    
    # Calculate slope and aspect
    slope, aspect = calculate_slope_aspect(dem_data, dem_transform, dem_crs)
    
    # Cache the results
    DEMCache.save_dem_cache(bbox, dem_data, dem_transform, dem_crs, slope, aspect)
    
    return dem_data, dem_transform, dem_crs, slope, aspect

def calculate_slope_aspect(dem: np.ndarray, transform, dem_crs) -> tuple:
    """Calculate slope and aspect from DEM.

    Properly handles geographic (lat/lon) vs projected CRS by converting
    pixel spacing to meters before computing gradients.
    """
    logger.debug("Calculating slope and aspect from DEM")

    # Extract pixel sizes from transform
    pixel_x = abs(transform.a)  # pixel width (may be degrees or meters)
    pixel_y = abs(transform.e)  # pixel height (may be degrees or meters)

    # Check if CRS is geographic (degrees) vs projected (meters)
    crs_obj = CRS.from_user_input(dem_crs) if dem_crs else None
    is_geographic = crs_obj.is_geographic if crs_obj else False

    if is_geographic:
        # Convert degrees to meters using latitude-aware calculation
        # Get center latitude of the DEM for conversion
        # transform.f is top-left Y, transform.e is negative pixel height
        center_lat = transform.f + (transform.e * dem.shape[0] / 2)

        # Meters per degree (approximate, varies with latitude)
        meters_per_degree_lat = 111320.0  # approximately constant
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(center_lat))

        # Convert pixel spacing from degrees to meters
        resolution_y = pixel_y * meters_per_degree_lat
        resolution_x = pixel_x * meters_per_degree_lon

        logger.debug(f"Geographic CRS detected at lat={center_lat:.2f}°")
        logger.debug(f"Pixel spacing: {pixel_y:.6f}° x {pixel_x:.6f}° = {resolution_y:.1f}m x {resolution_x:.1f}m")
    else:
        # Projected CRS - pixel spacing already in meters (or linear units)
        resolution_y = pixel_y
        resolution_x = pixel_x
        logger.debug(f"Projected CRS - pixel spacing: {resolution_y:.1f}m x {resolution_x:.1f}m")

    # Calculate gradients with correct spacing (dy uses Y spacing, dx uses X spacing)
    dy, dx = np.gradient(dem, resolution_y, resolution_x)

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

def _infer_pol_from_band_key(band_key: str) -> str:
    """Infer polarization name (vv or vh) from an asset/band key string"""
    key = (band_key or "").lower()
    if "vv" in key:
        return "vv"
    if "vh" in key:
        return "vh"
    return key

def parse_thermal_noise_xml(noise_url: str) -> dict | None:
    """Parse Sentinel-1 thermal noise XML to extract LUT.

    Returns a dict with the same schema as parse_calibration_xml:
    {'lines': [int], 'pixels': [[int]], 'values': [[float]]}
    """
    https_url = s3_to_https(noise_url)
    logger.debug(f"Fetching thermal noise XML from {https_url}")

    try:
        response = requests.get(https_url, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        # Prefer the range noise list for GRD
        list_specs = [
            ('.//noiseRangeVectorList', 'noiseRangeVector', ['noiseRangeLut', 'noiseLut', 'noiseLevel']),
            ('noiseRangeVectorList', 'noiseRangeVector', ['noiseRangeLut', 'noiseLut', 'noiseLevel']),
        ]

        selected_path = None
        noise_list = None
        vector_tag = None
        value_tags = None
        for path, vtag, vtags in list_specs:
            candidate = root.find(path)
            if candidate is not None:
                selected_path = path
                noise_list = candidate
                vector_tag = vtag
                value_tags = vtags
                break

        if noise_list is None:
            # Fall back to azimuth list if range not present (we will still try to use it)
            for path, vtag, vtags in [
                ('.//noiseAzimuthVectorList', 'noiseAzimuthVector', ['noiseAzimuthLut', 'noiseLut', 'noiseLevel']),
                ('noiseAzimuthVectorList', 'noiseAzimuthVector', ['noiseAzimuthLut', 'noiseLut', 'noiseLevel']),
            ]:
                candidate = root.find(path)
                if candidate is not None:
                    selected_path = path
                    noise_list = candidate
                    vector_tag = vtag
                    value_tags = vtags
                    break

        if noise_list is None:
            # Log available tags for debugging
            seen = set()
            for el in root.iter():
                seen.add(el.tag.split('}')[-1])
            logger.warning("No thermal noise data found in XML")
            logger.debug(f"Available tags: {sorted(seen)}")
            logger.debug(f"Noise XML head: {ET.tostring(root, encoding='unicode')[:800]}")
            return None

        logger.debug(f"Using noise list at path: {selected_path}, vectors tag: {vector_tag}")

        lut_data = {'lines': [], 'pixels': [], 'values': []}
        vectors = noise_list.findall(vector_tag)
        logger.debug(f"Found {len(vectors)} noise vectors in {vector_tag}")

        for idx, vector in enumerate(vectors):
            # Line (azimuth) index if present; otherwise fallback to sequential index
            line_el = vector.find('line')
            if line_el is not None and line_el.text:
                try:
                    line = int(float(line_el.text))
                except Exception:
                    line = idx
            else:
                line = idx

            # Pixels
            pixel_text = None
            for tag in ['pixel', 'pixels']:
                el = vector.find(tag)
                if el is not None and el.text:
                    pixel_text = el.text
                    break

            # Values
            value_el = None
            for tag in value_tags + ['noise', 'values']:
                el = vector.find(tag)
                if el is not None and el.text:
                    value_el = el
                    break

            # Fallback: derive pixels from first/last range sample and length of values
            if value_el is not None and pixel_text is None:
                fr = vector.find('firstRangeSample')
                lr = vector.find('lastRangeSample')
                try:
                    values_len = len(value_el.text.split())
                except Exception:
                    values_len = 0
                if fr is not None and lr is not None and fr.text and lr.text and values_len > 0:
                    try:
                        frv = int(float(fr.text))
                        lrv = int(float(lr.text))
                        if lrv >= frv:
                            step = max(1, (lrv - frv) // max(1, values_len - 1))
                            pixel_list_auto = list(range(frv, frv + step * values_len, step))[:values_len]
                            pixel_text = " ".join(str(p) for p in pixel_list_auto)
                    except Exception:
                        pass

            if pixel_text is None or value_el is None:
                    logger.debug("Skipping noise vector without pixel/value info")
                    continue

            try:
                pixel_list = [int(float(p)) for p in pixel_text.split()]
                value_list = [float(v) for v in value_el.text.split()]
            except Exception:
                logger.warning("Failed parsing noise vector entries; skipping vector")
                continue

            # Ensure lengths match
            if len(pixel_list) != len(value_list):
                logger.debug(f"Pixel/value length mismatch at line {line}: {len(pixel_list)} vs {len(value_list)}; aligning to min")
                n = min(len(pixel_list), len(value_list))
                pixel_list = pixel_list[:n]
                value_list = value_list[:n]

            lut_data['lines'].append(line)
            lut_data['pixels'].append(pixel_list)
            lut_data['values'].append(value_list)

        if not lut_data['lines']:
            logger.warning("Noise list present but contained no valid vectors")
            return None

        logger.debug(f"Parsed {len(lut_data['lines'])} noise lines; sample pixels: {len(lut_data['pixels'][0]) if lut_data['pixels'] else 0}")
        return lut_data
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch thermal noise XML: {e}")
        logger.debug(f"Original URL: {noise_url}")
        logger.debug(f"HTTPS URL: {https_url}")
        return None
    except Exception as e:
        logger.error(f"Failed to parse thermal noise XML: {e}")
        return None

def find_noise_asset(item, band_or_pol: str) -> str | None:
    """Find the thermal noise asset key for a given polarization.

    Prefers assets whose key or href matches 'schema-noise-<pol>.xml'.
    """
    pol = _infer_pol_from_band_key(band_or_pol)
    # Exact key candidates
    candidates = [
        f"schema-noise-{pol}.xml",
        f"noise-{pol}",
        f"{pol}-noise",
        f"noise_{pol}",
        f"{pol}_noise",
        "noise",  # generic
    ]

    for name in candidates:
        if name in item.assets:
            return name

    # Fallback: search by href filename or key contents
    for asset_name, asset in item.assets.items():
        name_l = asset_name.lower()
        href = getattr(asset, 'href', '') or ''
        href_l = href.lower()
        if pol in name_l and 'noise' in name_l:
            return asset_name
        if href_l.endswith(f"schema-noise-{pol}.xml"):
            return asset_name
        if pol in href_l and 'noise' in href_l and href_l.endswith('.xml'):
            return asset_name

    return None


def find_annotation_asset(item, band_or_pol: str) -> str | None:
    """Find the annotation asset key for a given polarization.

    Looks for annotation XML assets (iw-{pol}.xml pattern) that contain
    incidence angle grids. Excludes calibration and noise XMLs.
    """
    pol = _infer_pol_from_band_key(band_or_pol)

    # Try direct asset key patterns
    candidates = [
        f"iw-{pol}",
        f"{pol}-iw",
        f"iw{pol}",
        f"annotation-{pol}",
        f"{pol}-annotation",
    ]

    for name in candidates:
        if name in item.assets:
            href = getattr(item.assets[name], 'href', '') or ''
            href_l = href.lower()
            # Exclude calibration and noise XMLs
            if 'calibration' not in href_l and 'noise' not in href_l:
                logger.debug(f"Found annotation asset by key: {name}")
                return name

    # Fallback: search by href pattern
    for asset_name, asset in item.assets.items():
        href = getattr(asset, 'href', '') or ''
        href_l = href.lower()

        # Skip calibration and noise XMLs
        if 'calibration' in href_l or 'noise' in href_l:
            continue

        # Look for annotation XML with matching polarization
        if '/annotation/' in href_l and pol in href_l and href_l.endswith('.xml'):
            logger.debug(f"Found annotation asset by href pattern: {asset_name}")
            return asset_name

        # Match iw-{pol}.xml pattern in href
        if f"iw-{pol}.xml" in href_l or f"iw{pol}.xml" in href_l:
            logger.debug(f"Found annotation asset by filename: {asset_name}")
            return asset_name

    # Try constructing annotation URL from existing asset paths
    # The annotation file is typically in the same directory structure
    for asset_name in ['vv', 'vh', pol]:
        if asset_name in item.assets:
            base_href = item.assets[asset_name].href
            # Convert measurement path to annotation path
            # e.g., .../measurement/iw-vv.tiff -> .../annotation/iw-vv.xml
            if '/measurement/' in base_href:
                annotation_url = base_href.replace('/measurement/', '/annotation/')
                annotation_url = annotation_url.rsplit('.', 1)[0] + '.xml'
                # Return a synthetic key with the constructed URL
                logger.debug(f"Constructed annotation URL from {asset_name}: {annotation_url}")
                return f"__annotation_url__:{annotation_url}"

    return None


def get_annotation_href(item, annotation_key: str | None) -> str | None:
    """Get the annotation href from an asset key or synthetic key."""
    if annotation_key is None:
        return None

    # Handle synthetic keys (constructed URLs)
    if annotation_key.startswith("__annotation_url__:"):
        return annotation_key.split(":", 1)[1]

    # Regular asset key
    if annotation_key in item.assets:
        return item.assets[annotation_key].href

    return None


def parse_incidence_angle_grid(annotation_url: str) -> dict | None:
    """Parse Sentinel-1 annotation XML to extract incidence angle grid.

    Extracts the sparse geolocation grid (~10×21 points) with incidence angles
    that can be interpolated to full resolution.

    Returns:
        dict with keys:
            'lines': 1D array of unique azimuth line indices
            'pixels': 1D array of unique range pixel indices
            'angles': 2D array (n_lines × n_pixels) of incidence angles in degrees
            'mid_swath': incidenceAngleMidSwath value as fallback
        Returns None on error.
    """
    https_url = s3_to_https(annotation_url)
    logger.debug(f"Fetching annotation XML from {https_url}")

    try:
        response = requests.get(https_url, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        # Extract mid-swath incidence angle (scalar fallback)
        mid_swath = 35.0  # default
        mid_swath_el = root.find('.//imageAnnotation/imageInformation/incidenceAngleMidSwath')
        if mid_swath_el is not None and mid_swath_el.text:
            try:
                mid_swath = float(mid_swath_el.text)
            except ValueError:
                pass

        # Extract geolocation grid
        grid_list = root.find('.//geolocationGrid/geolocationGridPointList')
        if grid_list is None:
            logger.warning("No geolocation grid found in annotation XML, using mid-swath value")
            return {'lines': np.array([0]), 'pixels': np.array([0]),
                    'angles': np.array([[mid_swath]]), 'mid_swath': mid_swath}

        grid_points = grid_list.findall('geolocationGridPoint')
        if not grid_points:
            logger.warning("Empty geolocation grid in annotation XML")
            return {'lines': np.array([0]), 'pixels': np.array([0]),
                    'angles': np.array([[mid_swath]]), 'mid_swath': mid_swath}

        # Collect all grid points
        points_data = []
        for point in grid_points:
            try:
                line = int(point.find('line').text)
                pixel = int(point.find('pixel').text)
                inc_angle = float(point.find('incidenceAngle').text)
                points_data.append((line, pixel, inc_angle))
            except (AttributeError, ValueError, TypeError) as e:
                logger.debug(f"Skipping malformed grid point: {e}")
                continue

        if not points_data:
            logger.warning("No valid grid points parsed")
            return {'lines': np.array([0]), 'pixels': np.array([0]),
                    'angles': np.array([[mid_swath]]), 'mid_swath': mid_swath}

        # Get unique sorted line and pixel indices from points data
        all_lines = np.array([p[0] for p in points_data])
        all_pixels = np.array([p[1] for p in points_data])

        unique_lines = np.unique(all_lines)
        unique_pixels = np.unique(all_pixels)

        # Build 2D grid (n_lines × n_pixels)
        angle_grid = np.full((len(unique_lines), len(unique_pixels)), np.nan, dtype=np.float32)

        # Map indices
        line_to_idx = {line_val: i for i, line_val in enumerate(unique_lines)}
        pixel_to_idx = {pixel_val: i for i, pixel_val in enumerate(unique_pixels)}

        for line, pixel, angle in points_data:
            li = line_to_idx.get(line)
            pi = pixel_to_idx.get(pixel)
            if li is not None and pi is not None:
                angle_grid[li, pi] = angle

        # Log grid info
        valid_mask = ~np.isnan(angle_grid)
        if np.any(valid_mask):
            angle_min = np.nanmin(angle_grid)
            angle_max = np.nanmax(angle_grid)
            logger.debug(f"Parsed incidence angle grid: {len(unique_lines)}×{len(unique_pixels)} points, "
                        f"range: {angle_min:.2f}° to {angle_max:.2f}°")
        else:
            logger.warning("All grid points have NaN angles")

        return {
            'lines': unique_lines,
            'pixels': unique_pixels,
            'angles': angle_grid,
            'mid_swath': mid_swath,
        }

    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch annotation XML: {e}")
        return None
    except ET.ParseError as e:
        logger.warning(f"Failed to parse annotation XML: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error parsing annotation XML: {e}")
        return None


def interpolate_incidence_angle_to_image(grid_data: dict, shape: tuple) -> np.ndarray:
    """Interpolate sparse incidence angle grid to full image dimensions.

    Uses RegularGridInterpolator for bilinear interpolation with
    distance_transform_edt fallback for edge extrapolation.

    Args:
        grid_data: dict from parse_incidence_angle_grid() with keys:
            'lines', 'pixels', 'angles', 'mid_swath'
        shape: target (height, width) tuple

    Returns:
        float32 array of shape matching input, containing incidence angles in degrees
    """
    h, w = shape

    # Handle degenerate case (single point or no grid)
    if grid_data is None:
        logger.warning("No incidence angle grid data, using default 35°")
        return np.full((h, w), 35.0, dtype=np.float32)

    lines = grid_data['lines']
    pixels = grid_data['pixels']
    angles = grid_data['angles']
    mid_swath = grid_data.get('mid_swath', 35.0)

    # If grid is too small, just use mid-swath value
    if len(lines) < 2 or len(pixels) < 2:
        logger.debug(f"Grid too small ({len(lines)}×{len(pixels)}), using mid-swath: {mid_swath:.2f}°")
        return np.full((h, w), mid_swath, dtype=np.float32)

    # Fill any NaN values in the sparse grid with nearest neighbor before interpolation
    if np.any(np.isnan(angles)):
        mask = ~np.isnan(angles)
        if np.any(mask):
            indices = distance_transform_edt(~mask, return_indices=True)[1]
            angles = angles[tuple(indices)]
        else:
            logger.warning("All angles are NaN, using mid-swath")
            return np.full((h, w), mid_swath, dtype=np.float32)

    # Create interpolator with the sparse grid
    # Note: RegularGridInterpolator expects (row_coords, col_coords), data[row, col]
    try:
        interp = RegularGridInterpolator(
            (lines.astype(np.float64), pixels.astype(np.float64)),
            angles.astype(np.float64),
            method='linear',
            bounds_error=False,
            fill_value=None  # Extrapolate outside bounds
        )
    except ValueError as e:
        logger.warning(f"Failed to create interpolator: {e}, using mid-swath")
        return np.full((h, w), mid_swath, dtype=np.float32)

    # Create target grid coordinates
    target_lines = np.arange(h, dtype=np.float64)
    target_pixels = np.arange(w, dtype=np.float64)

    # Create meshgrid for all target points
    ll, pp = np.meshgrid(target_lines, target_pixels, indexing='ij')
    points = np.stack([ll.ravel(), pp.ravel()], axis=-1)

    # Interpolate
    interpolated = interp(points).reshape(h, w).astype(np.float32)

    # Handle any remaining NaN values from extrapolation using nearest neighbor
    nan_mask = np.isnan(interpolated)
    if np.any(nan_mask):
        valid_mask = ~nan_mask
        if np.any(valid_mask):
            indices = distance_transform_edt(nan_mask, return_indices=True)[1]
            interpolated = interpolated[tuple(indices)]
        else:
            logger.warning("All interpolated values are NaN, using mid-swath")
            interpolated = np.full((h, w), mid_swath, dtype=np.float32)

    angle_min = interpolated.min()
    angle_max = interpolated.max()
    logger.debug(f"Interpolated incidence angle to {h}×{w}, range: {angle_min:.2f}° to {angle_max:.2f}°")

    return interpolated


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

def apply_radiometric_calibration(dn_data: np.ndarray, calibration_lut: np.ndarray, noise_lut: np.ndarray | None = None) -> np.ndarray:
    """Apply thermal noise removal and radiometric calibration to convert DN to gamma0.

    Steps:
    - Convert DN to power (DN^2)
    - Subtract thermal noise power LUT if provided
    - Divide by calibration LUT squared to obtain gamma0
    """
    p0, p25, p50, p75, p100 = np.percentile(dn_data, [0, 25, 50, 75, 100])
    logger.debug(f"DN percentiles: {p0:.2f}, {p25:.2f}, {p50:.2f}, {p75:.2f}, {p100:.2f}")
    logger.debug(f"Applying noise removal + calibration - DN range: {dn_data.min():.1f} to {dn_data.max():.1f}")

    dn_power = dn_data.astype(np.float32) ** 2
    if noise_lut is not None:
        logger.debug("Subtracting thermal noise from power")
        before_p = None
        try:
            v = dn_power[dn_power > 0]
            if v.size:
                before_p = np.percentile(v, [5, 50, 95])
        except Exception:
            pass
        dn_power_sub = dn_power - noise_lut.astype(np.float32)
        clipped = dn_power_sub <= 0
        clipped_ratio = float(np.count_nonzero(clipped)) / dn_power_sub.size
        dn_power = np.where(clipped, 0.0, dn_power_sub)
        try:
            v2 = dn_power[dn_power > 0]
            if v2.size:
                after_p = np.percentile(v2, [5, 50, 95])
                if before_p is not None:
                    logger.debug(f"DN power p5/50/95 before: {before_p[0]:.3g}/{before_p[1]:.3g}/{before_p[2]:.3g}; after noise: {after_p[0]:.3g}/{after_p[1]:.3g}/{after_p[2]:.3g}; clipped={clipped_ratio:.1%}")
                else:
                    logger.debug(f"DN power after noise p5/50/95: {after_p[0]:.3g}/{after_p[1]:.3g}/{after_p[2]:.3g}; clipped={clipped_ratio:.1%}")
            else:
                logger.debug(f"All pixels clipped by noise subtraction; clipped={clipped_ratio:.1%}")
        except Exception:
            pass

    with np.errstate(divide='ignore', invalid='ignore'):
        gamma0 = dn_power / (calibration_lut.astype(np.float32) ** 2)
        gamma0 = np.where(calibration_lut > 0, gamma0, 0)
        p0, p25, p50, p75, p100 = np.percentile(gamma0[gamma0 > 0], [0, 25, 50, 75, 100]) if np.any(gamma0 > 0) else (0,0,0,0,0)
        logger.debug(f"Gamma0 percentiles: {p0:.2f}, {p25:.2f}, {p50:.2f}, {p75:.2f}, {p100:.2f}")

    valid_mask = gamma0 > 0
    if np.any(valid_mask):
        logger.debug(f"Gamma0 range: {gamma0[valid_mask].min():.2f} to {gamma0[valid_mask].max():.2f}")
    else:
        logger.warning("No valid gamma0 values after calibration")

    return gamma0

def apply_terrain_flattening(gamma0: np.ndarray, slope: np.ndarray, aspect: np.ndarray,
                            incidence_angle: float | np.ndarray, look_direction: float) -> np.ndarray:
    """Apply terrain flattening to convert gamma0 to gamma0_terrain.

    Args:
        gamma0: Radiometrically calibrated backscatter
        slope: Terrain slope in radians
        aspect: Terrain aspect in radians
        incidence_angle: Incidence angle in degrees (scalar or array matching gamma0 shape)
        look_direction: Satellite look direction in radians

    Returns:
        Terrain-flattened gamma0 array
    """
    # Log incidence angle info
    if isinstance(incidence_angle, np.ndarray):
        angle_min = incidence_angle.min()
        angle_max = incidence_angle.max()
        logger.debug(f"Applying terrain flattening (incidence array: {angle_min:.2f}° to {angle_max:.2f}°)")
    else:
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

def find_calibration_asset(item, band: str) -> str | None:
    """Find the correct calibration asset name for a band.

    Uses polarization-aware matching to avoid returning VH calibration for VV or vice versa.
    """
    # Get polarization from band name
    pol = _infer_pol_from_band_key(band)

    # List all available assets for debugging
    logger.debug(f"Available assets for {item.id}: {list(item.assets.keys())}")
    logger.debug(f"Looking for calibration for polarization: {pol}")

    # Try polarization-specific naming conventions first
    possible_names = [
        f"{pol}-calibration",
        f"calibration-{pol}",
        f"{pol}_calibration",
        f"calibration_{pol}",
        f"schema-calibration-{pol}.xml",
        f"{pol}-cal",
        f"cal-{pol}",
    ]

    for name in possible_names:
        if name in item.assets:
            logger.debug(f"Found calibration asset: {name}")
            return name

    # Fallback: search by asset name or href, but REQUIRE polarization match
    for asset_name, asset in item.assets.items():
        name_l = asset_name.lower()
        href = getattr(asset, 'href', '') or ''
        href_l = href.lower()

        # Only return if BOTH polarization AND calibration keywords are present
        if pol in name_l and ('calibration' in name_l or 'cal' in name_l):
            logger.debug(f"Found calibration asset by name match: {asset_name}")
            return asset_name

        # Check href for calibration files with matching polarization
        if href_l.endswith(f"schema-calibration-{pol}.xml"):
            logger.debug(f"Found calibration asset by href match: {asset_name}")
            return asset_name
        if pol in href_l and ('calibration' in href_l or '/cal' in href_l) and href_l.endswith('.xml'):
            logger.debug(f"Found calibration asset by href content match: {asset_name}")
            return asset_name

    logger.warning(f"No calibration asset found for polarization {pol}")
    return None

def process_band_with_terrain_correction(
    data_href: str,
    calibration_href: str,
    noise_href: str | None,
    annotation_href: str | None,
    bounds: tuple,
    target_crs: str,
    dem_data: np.ndarray,
    dem_transform,
    dem_crs,
    slope: np.ndarray,
    aspect: np.ndarray,
    incidence_angle_fallback: float,
    look_direction: float,
    aws_session,
    simplified_incidence: bool = False,
) -> np.ndarray:
    """Process a single band with calibration and terrain correction.

    Args:
        data_href: URL to the SAR data
        calibration_href: URL to the calibration XML
        noise_href: URL to the thermal noise XML (optional)
        annotation_href: URL to the annotation XML for incidence angle grid (optional)
        bounds: Target bounding box
        target_crs: Target CRS
        dem_data: DEM array
        dem_transform: DEM transform
        dem_crs: DEM CRS
        slope: Terrain slope array
        aspect: Terrain aspect array
        incidence_angle_fallback: Fallback incidence angle from STAC properties (degrees)
        look_direction: Satellite look direction (radians)
        aws_session: AWS session for rasterio
        simplified_incidence: If True, use mid-swath scalar instead of interpolated grid
    """

    logger.debug("Processing band with terrain correction")
    logger.debug(f"Data URL: {data_href}")
    logger.debug(f"Calibration URL: {calibration_href}")
    logger.debug(f"Noise URL: {noise_href}")
    logger.debug(f"Annotation URL: {annotation_href}")
    
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

    # Determine incidence angle (array or scalar)
    incidence_angle: float | np.ndarray = incidence_angle_fallback

    if annotation_href:
        grid_data = parse_incidence_angle_grid(annotation_href)
        if grid_data is not None:
            if simplified_incidence:
                # Use mid-swath scalar value
                incidence_angle = grid_data.get('mid_swath', incidence_angle_fallback)
                logger.debug(f"Using mid-swath incidence angle: {incidence_angle:.2f}°")
            else:
                # Interpolate to full image size
                incidence_angle = interpolate_incidence_angle_to_image(grid_data, sar_shape)
                logger.debug(f"Using incidence angle from annotation XML grid: "
                            f"{incidence_angle.min():.2f}° to {incidence_angle.max():.2f}°")
        else:
            logger.debug(f"Failed to parse annotation, using STAC/fallback incidence angle: {incidence_angle_fallback:.1f}°")
    else:
        logger.debug(f"No annotation URL, using STAC/fallback incidence angle: {incidence_angle_fallback:.1f}°")

    # Parse and apply thermal noise + calibration
    noise_lut = None
    if noise_href:
        noise_lut_data = parse_thermal_noise_xml(noise_href)
        if noise_lut_data:
            noise_lut = interpolate_lut_to_image(noise_lut_data, sar_shape)
            try:
                nz = noise_lut[noise_lut > 0]
                if nz.size:
                    p0, p50, p95 = np.percentile(nz, [0, 50, 95])
                    logger.debug(f"Noise LUT stats: min={nz.min():.3g}, med={p50:.3g}, p95={p95:.3g}, max={nz.max():.3g}")
                else:
                    logger.debug("Noise LUT contains no positive values")
            except Exception:
                pass
        else:
            logger.warning("Failed to parse noise XML; proceeding without noise removal")

    if calibration_href:
        lut_data = parse_calibration_xml(calibration_href, 'gamma')
        if lut_data:
            calibration_lut = interpolate_lut_to_image(lut_data, sar_shape)
            # For debugging: show DN power vs noise percentile before subtraction
            try:
                if noise_lut is not None:
                    dn_power = dn_data.astype(np.float32) ** 2
                    dn_valid = dn_power[dn_power > 0]
                    noise_valid = noise_lut[noise_lut > 0]
                    if dn_valid.size and noise_valid.size:
                        dp50 = np.percentile(dn_valid, 50)
                        np50 = np.percentile(noise_valid, 50)
                        logger.debug(f"DN power median={dp50:.3g}, noise median={np50:.3g}")
            except Exception:
                pass
            gamma0 = apply_radiometric_calibration(dn_data, calibration_lut, noise_lut)
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
                resampling=Resampling.nearest
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
        # Compute percentiles for diagnostics and optional scaling
        p1, p99 = np.percentile(gamma0_terrain[valid_mask], [0.1, 99.9])
        p0, p25, p50, p75, p100 = np.percentile(gamma0_terrain[valid_mask], [0, 25, 50, 75, 100])
        logger.debug(f"Gamma0 terrain percentiles: p0={p0:.4f}, p25={p25:.4f}, p50={p50:.4f}, p75={p75:.4f}, p100={p100:.4f}")
        logger.debug(f"Gamma0 terrain p0.1={p1:.4f}, p99.9={p99:.4f}")

        # =======================================================================
        # Temporary debugging - looking into impact of scaling options:
        # =======================================================================

        #   | Option      | Method                   | Best For                               |
        #   |-------------|--------------------------|----------------------------------------|
        #   | A (current) | Clip [0, 1]              | May clip valid data? (gamma0 can > 1)  |
        #   | B           | Percentile normalization | Maximizing dynamic range per scene     |
        #   | C           | Linear [0, 2]            | Consistent scaling, most land surfaces |
        #   | D           | Linear [0, 3]            | Urban areas, strong scatterers         |
        #   | E           | dB [-30, +10]            | Standard SAR, wide dynamic range       |
        #   | F           | dB [-25, +5]             | Vegetation/agriculture focus           |

        # OPTION A: Current - Hard clip at [0, 1] (PROBLEMATIC: gamma0 can exceed 1)
        # Use case: Only valid if gamma0 is always < 1 (rarely true)
        # Problem: Clips high backscatter from urban areas, corner reflectors
        scaled = np.clip(gamma0_terrain, 0, 1) * 65535

        # OPTION B: Percentile-based normalization (preserves relative differences)
        # Use case: Maximizes dynamic range, good for visualization
        # Problem: Absolute values are lost, varies per scene
        # scaled = np.clip((gamma0_terrain - p1) / (p99 - p1 + 1e-10), 0, 1) * 65535

        # OPTION C: Fixed linear range [0, 2] (gamma0 rarely exceeds 2)
        # Use case: Consistent scaling across scenes, preserves most data
        # Problem: Very high backscatter (>2) still clipped
        # scaled = np.clip(gamma0_terrain / 2.0, 0, 1) * 65535

        # OPTION D: Fixed linear range [0, 3] (more headroom for bright targets)
        # Use case: Urban areas, strong scatterers
        # Problem: Reduces precision for typical vegetation/soil values
        # scaled = np.clip(gamma0_terrain / 3.0, 0, 1) * 65535

        # OPTION E: dB scale - standard for SAR, maps -30 to +10 dB
        # Use case: Standard SAR visualization, compresses dynamic range naturally
        # Problem: Requires knowing the dB range of interest
        # gamma0_db = 10 * np.log10(np.where(gamma0_terrain > 1e-10, gamma0_terrain, 1e-10))
        # scaled = np.clip((gamma0_db + 30) / 40, 0, 1) * 65535  # -30dB->0, +10dB->65535

        # OPTION F: dB scale - narrower range -25 to +5 dB (typical land surfaces)
        # Use case: Better precision for vegetation/agriculture
        # gamma0_db = 10 * np.log10(np.where(gamma0_terrain > 1e-10, gamma0_terrain, 1e-10))
        # scaled = np.clip((gamma0_db + 25) / 30, 0, 1) * 65535  # -25dB->0, +5dB->65535

        # =======================================================================

        logger.debug("Scaling method: OPTION A (clip 0-1)")
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
    
    logger.debug("Processing band (simple mode)")
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

def get_quarterly_scenes_by_coverage(items: list, year: int, tile_bounds: tuple,
                                     coverage_threshold: float = 0.95,
                                     k_scenes: int = 1) -> tuple:
    """Select up to k_scenes per quarter, preferring items whose footprint covers the tile.

    Returns (selected_items_dict, per_quarter_metadata_list)
    where selected_items_dict[quarter] is a list of items (length <= k_scenes)
    """
    quarters = {
        'Q1': (f'{year}-01-15', f'{year}-03-15'),
        'Q2': (f'{year}-04-15', f'{year}-06-15'),
        'Q3': (f'{year}-07-15', f'{year}-09-15'),
        'Q4': (f'{year}-10-15', f'{year}-12-15')
    }

    selected = {}
    quarter_meta = []

    for q_name, (start, end) in quarters.items():
        start_dt = np.datetime64(start)
        end_dt = np.datetime64(end)

        quarter_items = []
        for item in items:
            item_dt = np.datetime64(item.properties["datetime"])
            if start_dt <= item_dt <= end_dt:
                quarter_items.append(item)

        if not quarter_items:
            logger.warning(f"{q_name}: No scenes found")
            selected[q_name] = []
            quarter_meta.append({
                "quarter": q_name,
                "candidate_count": 0,
                "selected_ids": [],
                "selected_datetimes": [],
                "selected_orbit": None,
                "coverage_fractions": [],
                "k_selected": 0,
            })
            continue

        # Score by coverage fraction
        scored = []
        for it in quarter_items:
            cf = coverage_fraction(it, tile_bounds)
            scored.append((cf, it))

        scored.sort(key=lambda t: t[0], reverse=True)

        # Select top k_scenes that meet coverage threshold (or best available)
        top_k = []
        for cf, it in scored[:k_scenes]:
            if cf >= coverage_threshold or len(top_k) == 0:
                top_k.append((cf, it))
            else:
                # Below threshold and we already have at least one scene
                break

        # If we didn't get any above threshold, take the best one anyway
        if not top_k:
            top_k = [scored[0]]

        selected[q_name] = [it for cf, it in top_k]
        coverages = [cf for cf, it in top_k]

        logger.info(f"{q_name}: Selected {len(top_k)} scene(s) with coverages {[f'{c:.3f}' for c in coverages]}")

        quarter_meta.append({
            "quarter": q_name,
            "candidate_count": len(scored),
            "selected_ids": [it.id for cf, it in top_k],
            "selected_datetimes": [it.properties.get("datetime") for cf, it in top_k],
            "selected_orbit": top_k[0][1].properties.get('sat:orbit_state') if top_k else None,
            "coverage_fractions": coverages,
            "k_selected": len(top_k),
        })

    return selected, quarter_meta


def composite_quarter_scenes(scene_arrays: list, method: str = 'mean') -> np.ndarray:
    """Composite multiple scene arrays into a single array using specified method.

    Args:
        scene_arrays: List of arrays, each with shape (bands, H, W)
        method: 'median' or 'mean'

    Returns:
        Composited array with shape (bands, H, W)
    """
    if not scene_arrays:
        raise ValueError("No scenes to composite")

    if len(scene_arrays) == 1:
        return scene_arrays[0]

    # Stack all scenes: (k, bands, H, W)
    stacked = np.stack(scene_arrays, axis=0).astype(np.float32)

    # Replace zeros with NaN for proper median/mean calculation
    stacked = np.where(stacked == 0, np.nan, stacked)

    # Compute composite
    with np.errstate(all='ignore'):
        if method == 'median':
            composited = np.nanmedian(stacked, axis=0)
        elif method == 'mean':
            composited = np.nanmean(stacked, axis=0)
        else:
            raise ValueError(f"Unknown compositing method: {method}")

    # Replace NaN with 0
    composited = np.where(np.isnan(composited), 0, composited)

    logger.debug(f"Composited {len(scene_arrays)} scenes using {method}, shape: {composited.shape}")

    return composited.astype(np.uint16)

def main():
    ap = argparse.ArgumentParser(description="Load Sentinel-1 data with terrain correction (quarterly, obstore).")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--X_tile", type=int, required=True)
    ap.add_argument("--Y_tile", type=int, required=True)
    ap.add_argument("--dest", type=str, required=True)
    ap.add_argument("--expansion", type=int, default=300)
    ap.add_argument("--coverage-threshold", type=float, default=0.95, help="Minimum tile coverage fraction to prefer a scene")
    ap.add_argument("--no-terrain-correction", action="store_true", help="Disable terrain correction")
    ap.add_argument("--no-calibration", action="store_true", help="Disable radiometric calibration")
    ap.add_argument("--simplified-incidence", action="store_true",
                    help="Use mid-swath incidence angle only (faster, less accurate)")
    ap.add_argument("--orbit-direction", type=str, choices=["ASCENDING", "DESCENDING", "BOTH"], default="ASCENDING",
                    help="Filter by orbit direction (default: ASCENDING)")
    ap.add_argument("--k-scenes", type=int, default=3, help="Number of scenes to composite per quarter (default: 3)")
    ap.add_argument("--debug", action="store_true", help="Enable debug logging")
    ap.add_argument("--run-metadata", action="store_true", help="Write metadata sidecar file")
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
        store = from_url(args.dest, region = "us-east-1")
    #if isinstance(store, LocalStore):
    else:
        os.makedirs(args.dest, exist_ok=True)
        #os.makedirs(store.prefix, exist_ok=True)
        store = LocalStore(prefix=args.dest)
    
    base_key = f"{args.year}/raw/{args.X_tile}/{args.Y_tile}/raw"
    s1_dir_key = f"{base_key}/s1"
    misc_key = f"{base_key}/misc"
    #ensure_dirs(store, s1_dir_key, misc_key)
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

    url = search.url_with_parameters()
    logger.debug(f"STAC URL: {url}")

    items = list(search.items())
    if not items:
        raise RuntimeError("No Sentinel-1 items found for query.")

    logger.info(f"Found {len(items)} Sentinel-1 scenes for {args.year}")

    # Filter by orbit direction
    if args.orbit_direction != "BOTH":
        items = [
            item for item in items
            if item.properties.get("sat:orbit_state", "").upper() == args.orbit_direction
        ]
        logger.info(f"Filtered to {len(items)} scenes for {args.orbit_direction} orbit")
        if not items:
            raise RuntimeError(f"No Sentinel-1 items found for {args.orbit_direction} orbit.")

    # Get DEM data if terrain correction is enabled
    dem_data = dem_transform = dem_crs = slope = aspect = None
    if not args.no_terrain_correction:
        dem_data, dem_transform, dem_crs, slope, aspect = fetch_copdem_for_bbox(bbx)
        if dem_data is None:
            logger.warning("DEM fetch failed, proceeding without terrain correction")
        logger.debug(f"Slope stats: min={slope.min()}, max={slope.max()}, mean={slope.mean()}, std={slope.std()}")
        logger.debug(f"Aspect stats: min={aspect.min()}, max={aspect.max()}, mean={aspect.mean()}, std={aspect.std()}")

    # Compute tile bounds once
    tile_bounds = featureBounds(bbox2geojson(bbx))

    # Select quarterly scenes with coverage filtering
    quarterly_items, quarter_meta = get_quarterly_scenes_by_coverage(
        items, args.year, tile_bounds, args.coverage_threshold, args.k_scenes
    )
    
    # Determine which polarizations exist
    akeys = set()
    for item_list in quarterly_items.values():
        for item in item_list:
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
    else:
        # Canonicalize ordering to ['vv','vh'] where present
        bands = [b for b in ASSETS_S1 if b in bands]

    logger.info(f"Processing bands: {bands}")

    target_crs = "EPSG:4326"
    bounds = tile_bounds

    # Process each quarter with multi-scene compositing
    quarterly_arrays = []
    quarter_band_stats = {q: {} for q in ['Q1','Q2','Q3','Q4']}

    def process_single_item(item, bands, bounds, target_crs, dem_data, dem_transform, dem_crs,
                            slope, aspect, aws_session, no_calibration, no_terrain_correction,
                            simplified_incidence=False):
        """Process a single STAC item and return array of shape (bands, H, W)."""
        # Get geometry for this scene (fallback incidence angle from STAC properties)
        incidence_angle_fallback, look_direction = get_item_geometry(item)

        # Process each band
        band_arrays = []
        for band in bands:
            logger.debug(f"  Looking for band: {band}")

            # Check if band exists in assets
            actual_band = band
            if band not in item.assets:
                logger.warning(f"    Band {band} not found in assets")
                # Try alternative naming
                found = False
                for asset_name in item.assets:
                    if band in asset_name.lower():
                        logger.info(f"    Using alternative asset name: {asset_name} for band {band}")
                        actual_band = asset_name
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
            if actual_band in item.assets:
                logger.debug(f"    Processing band {actual_band}")

                # Try to find calibration asset - use the original band name (vv or vh)
                cal_asset_name = None
                if not no_calibration:
                    cal_asset_name = find_calibration_asset(item, band)
                # Try to find thermal noise asset per polarization
                noise_asset_name = find_noise_asset(item, band)
                # Try to find annotation asset for incidence angle grid
                annotation_asset_name = find_annotation_asset(item, band)

                if cal_asset_name and not no_calibration:
                    cal_href = item.assets[cal_asset_name].href
                    logger.debug(f"    Using calibration from: {cal_asset_name}")
                else:
                    cal_href = None
                    if not no_calibration:
                        logger.warning(f"    No calibration found for {band}")
                if noise_asset_name:
                    noise_href = item.assets[noise_asset_name].href
                    logger.debug(f"    Using thermal noise from: {noise_asset_name}")
                else:
                    noise_href = None
                # Get annotation href (handles both regular and synthetic keys)
                annotation_href = get_annotation_href(item, annotation_asset_name)
                if annotation_href:
                    logger.debug(f"    Using annotation from: {annotation_asset_name}")
                else:
                    logger.debug(f"    No annotation asset found for {band}")

                if dem_data is not None and not no_terrain_correction and cal_href:
                    # Full processing with calibration and terrain correction
                    arr = process_band_with_terrain_correction(
                        item.assets[actual_band].href,
                        cal_href,
                        noise_href,
                        annotation_href,
                        bounds,
                        target_crs,
                        dem_data, dem_transform, dem_crs,
                        slope, aspect,
                        incidence_angle_fallback, look_direction,
                        aws_session,
                        simplified_incidence=simplified_incidence,
                    )
                else:
                    # Simple processing
                    arr = process_band_simple(
                        item.assets[actual_band].href,
                        bounds,
                        target_crs,
                        aws_session
                    )

                band_arrays.append(arr)
            else:
                logger.warning(f"    Band {actual_band} not available")
                if band_arrays:
                    shape = band_arrays[0].shape
                else:
                    shape = (512, 512)
                band_arrays.append(np.zeros(shape, dtype=np.uint16))

        if band_arrays:
            return np.stack(band_arrays, axis=0)
        else:
            return None

    for q_name in ['Q1', 'Q2', 'Q3', 'Q4']:
        item_list = quarterly_items.get(q_name, [])

        if not item_list:
            logger.warning(f"No data for {q_name}, using zeros")
            # Create dummy data matching expected shape
            if quarterly_arrays:
                h, w = quarterly_arrays[0].shape[1:3]
            else:
                h, w = 512, 512  # Default size
            quarter_data = np.zeros((len(bands), h, w), dtype=np.uint16)
        else:
            logger.info(f"Processing {q_name}: {len(item_list)} scene(s)")

            # Process each scene in this quarter
            scene_arrays = []
            for idx, item in enumerate(item_list):
                logger.info(f"  Processing scene {idx+1}/{len(item_list)}: {item.id}")
                logger.debug(f"    Available assets: {list(item.assets.keys())}")

                scene_data = process_single_item(
                    item, bands, bounds, target_crs,
                    dem_data, dem_transform, dem_crs, slope, aspect,
                    aws_session, args.no_calibration, args.no_terrain_correction,
                    simplified_incidence=args.simplified_incidence,
                )

                if scene_data is not None:
                    scene_arrays.append(scene_data)
                    logger.debug(f"    Scene data shape: {scene_data.shape}, range: {scene_data.min()} to {scene_data.max()}")

            # Composite multiple scenes if we have more than one
            if scene_arrays:
                if len(scene_arrays) > 1:
                    # Ensure all arrays have the same shape
                    max_h = max(a.shape[1] for a in scene_arrays)
                    max_w = max(a.shape[2] for a in scene_arrays)
                    padded_arrays = []
                    for arr in scene_arrays:
                        if arr.shape[1] < max_h or arr.shape[2] < max_w:
                            pad_h = max_h - arr.shape[1]
                            pad_w = max_w - arr.shape[2]
                            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')
                        padded_arrays.append(arr)
                    quarter_data = composite_quarter_scenes(padded_arrays, method='median')
                    logger.info(f"  Composited {len(scene_arrays)} scenes for {q_name}")
                else:
                    quarter_data = scene_arrays[0]
                logger.debug(f"  Quarter data shape: {quarter_data.shape}, range: {quarter_data.min()} to {quarter_data.max()}")
            else:
                logger.error(f"  No scenes processed for {q_name}")
                quarter_data = np.zeros((len(bands), 512, 512), dtype=np.uint16)
        
        # Compute per-band stats for this quarter
        try:
            for bi, bname in enumerate(bands):
                quarter_band_stats[q_name][bname] = compute_band_stats(quarter_data[bi])
        except Exception as e:
            logger.warning(f"Failed computing band stats for {q_name}: {e}")

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

    # Compute final per-band stats across all timesteps
    final_band_stats = {}
    try:
        for bi, bname in enumerate(bands):
            final_band_stats[bname] = compute_band_stats(final_array[..., bi])
    except Exception as e:
        logger.warning(f"Failed computing final band stats: {e}")
    
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

    # Sidecar metadata file
    if args.run_metadata:
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            meta_key = f"{misc_key}/s1_meta_{args.X_tile}X{args.Y_tile}Y_{ts}.txt"
            lines = []
            lines.append("Sentinel-1 quarterly processing metadata")
            lines.append(f"timestamp: {ts}")
            lines.append(f"year: {args.year}")
            lines.append(f"lon: {args.lon}")
            lines.append(f"lat: {args.lat}")
            lines.append(f"X_tile: {args.X_tile}")
            lines.append(f"Y_tile: {args.Y_tile}")
            lines.append(f"bbox: {bbx}")
            lines.append(f"bounds: {bounds}")
            lines.append(f"items_found: {len(items)}")
            lines.append(f"coverage_threshold: {args.coverage_threshold}")
            lines.append(f"orbit_direction: {args.orbit_direction}")
            lines.append(f"k_scenes: {args.k_scenes}")
            lines.append(f"bands: {bands}")
            lines.append(f"calibration_enabled: {not args.no_calibration}")
            lines.append(f"terrain_correction_enabled: {not args.no_terrain_correction}")
            if dem_data is None:
                lines.append("dem: none")
            else:
                lines.append("dem: present")
            lines.append("quarters:")
            for qm in quarter_meta:
                lines.append(f"  - quarter: {qm['quarter']}")
                lines.append(f"    k_selected: {qm['k_selected']}")
                lines.append(f"    selected_ids: {qm['selected_ids']}")
                lines.append(f"    selected_datetimes: {qm['selected_datetimes']}")
                lines.append(f"    orbit: {qm['selected_orbit']}")
                lines.append(f"    coverage_fractions: {qm['coverage_fractions']}")
                lines.append(f"    candidate_count: {qm['candidate_count']}")
            lines.append(f"output_shape: {list(final_array.shape)}")
            lines.append(f"output_dtype: {str(final_array.dtype)}")
            lines.append(f"output_min: {int(final_array.min())}")
            lines.append(f"output_max: {int(final_array.max())}")
            # Per-quarter band stats
            lines.append("quarter_band_stats:")
            for qn in ['Q1','Q2','Q3','Q4']:
                lines.append(f"  {qn}:")
                qb = quarter_band_stats.get(qn, {})
                for bname in bands:
                    stats = qb.get(bname, None)
                    if stats is None:
                        continue
                    lines.append(f"    {bname}:")
                    lines.append(f"      min: {stats['min']}")
                    lines.append(f"      max: {stats['max']}")
                    lines.append(f"      mean: {stats['mean']:.3f}")
                    lines.append(f"      std: {stats['std']:.3f}")
                    lines.append(f"      p5: {stats['p5']:.3f}")
                    lines.append(f"      p50: {stats['p50']:.3f}")
                    lines.append(f"      p95: {stats['p95']:.3f}")
                    lines.append(f"      valid_ratio: {stats['valid_ratio']:.4f}")
                    lines.append(f"      count: {stats['count']}")
                    lines.append(f"      valid_count: {stats['valid_count']}")
            # Final band stats
            lines.append("final_band_stats:")
            for bname in bands:
                stats = final_band_stats.get(bname, None)
                if stats is None:
                    continue
                lines.append(f"  {bname}:")
                lines.append(f"    min: {stats['min']}")
                lines.append(f"    max: {stats['max']}")
                lines.append(f"    mean: {stats['mean']:.3f}")
                lines.append(f"    std: {stats['std']:.3f}")
                lines.append(f"    p5: {stats['p5']:.3f}")
                lines.append(f"    p50: {stats['p50']:.3f}")
                lines.append(f"    p95: {stats['p95']:.3f}")
                lines.append(f"    valid_ratio: {stats['valid_ratio']:.4f}")
                lines.append(f"    count: {stats['count']}")
                lines.append(f"    valid_count: {stats['valid_count']}")
            lines.append(f"s1_key: {fn_s1_key}")
            lines.append(f"dates_key: {fn_s1_dates_key}")
            lines.append(f"full_s1_path: {full_s1_path}")
            lines.append(f"full_dates_path: {full_dates_path}")
            lines.append(f"write_ms: {_elapsed_ms(t_write):.0f}")
            lines.append(f"total_ms: {_elapsed_ms(t_all):.0f}")
            obstore_put_text(store, meta_key, "\n".join(lines) + "\n")
            logger.success(f"S1 metadata sidecar saved: {args.dest.rstrip('/')}/{meta_key}")
        except Exception as e:
            logger.error(f"Failed to write metadata sidecar: {e}")

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
    run_metadata: bool = False,
    orbit_direction: str = "ASCENDING",
    k_scenes: int = 3,
    simplified_incidence: bool = False,
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
        "--orbit-direction", orbit_direction,
        "--k-scenes", str(k_scenes),
    ]
    if no_terrain_correction:
        argv.append("--no-terrain-correction")
    if no_calibration:
        argv.append("--no-calibration")
    if simplified_incidence:
        argv.append("--simplified-incidence")
    if debug:
        argv.append("--debug")
    if run_metadata:
        argv.append("--run-metadata")

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
        "simplified_incidence": bool(simplified_incidence),
        "run_metadata": bool(run_metadata),
        "orbit_direction": orbit_direction,
        "k_scenes": k_scenes,
    }


if __name__ == "__main__":
    main()