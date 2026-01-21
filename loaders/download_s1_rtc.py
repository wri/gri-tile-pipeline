#!/usr/bin/env python3
"""
Planetary Computer Sentinel-1 RTC quarterly composites (VV/VH) -> HKL.

Emulates original outputs:
  /{YEAR}/raw/{X_tile}/{Y_tile}/raw/misc/s1_dates_{X_tile}X{Y_tile}Y.hkl
  /{YEAR}/raw/{X_tile}/{Y_tile}/raw/s1/{X_tile}X{Y_tile}Y.hkl

Produces:
  raw/s1/{X}X{Y}Y.hkl           (uint16, (12,H,W,2))  # 4 quarters x 3 repeats = 12
  raw/misc/s1_dates_{tile}.hkl  (int64, (12,))        # [45,45,45,135,135,135,225,225,225,315,315,315]


"""

import os
import sys
import argparse
import tempfile
import time
import json
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

from datetime import datetime

import numpy as np
import rasterio as rio
# from rasterio.vrt import WarpedVRT
# from rasterio.enums import Resampling
# from rasterio.windows import from_bounds
from rasterio.features import bounds as featureBounds

from pystac_client import Client
import planetary_computer as pc
from shapely.geometry import shape, box

import hickle as hkl
from loguru import logger

from odc.stac import load as stac_load, configure_rio

import obstore as obs
from obstore.store import LocalStore, from_url
import random
import traceback
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar('T')

# -------------------- Custom Exceptions --------------------
class S1ProcessingError(Exception):
    """Base exception for S1 processing errors."""
    pass


class RetryableError(S1ProcessingError):
    """Errors that may succeed on retry (429, 5xx, timeout, network)."""
    def __init__(self, message: str, original_exception: Exception = None,
                 attempt: int = 0, operation: str = ""):
        super().__init__(message)
        self.original_exception = original_exception
        self.attempt = attempt
        self.operation = operation


class PermanentError(S1ProcessingError):
    """Errors that will not succeed on retry (bad geometry, no data, auth)."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception


# -------------------- Retry Utilities --------------------
def _is_retryable_exception(exc: Exception) -> bool:
    """Classify exception as retryable based on type/message."""
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    # HTTP status codes indicating transient issues
    retryable_codes = ['429', '500', '502', '503', '504', '408']
    if any(code in exc_str for code in retryable_codes):
        return True

    # Network/timeout keywords
    retryable_keywords = [
        'timeout', 'timed out', 'connection', 'reset', 'refused',
        'temporary', 'throttl', 'rate limit', 'too many requests',
        'service unavailable', 'bad gateway', 'internal server error',
        'network', 'socket', 'ssl', 'eof'
    ]
    if any(kw in exc_str for kw in retryable_keywords):
        return True

    # Known retryable exception types
    retryable_types = ['timeout', 'connection', 'http', 'urllib', 'ssl']
    if any(rt in exc_type for rt in retryable_types):
        return True

    return False


def retry_with_backoff(
    operation: Callable[[], T],
    operation_name: str,
    max_attempts: int = 5,
    initial_backoff: float = 0.5,
    max_backoff: float = 30.0,
    backoff_factor: float = 2.0,
    jitter_range: tuple = (0.0, 1.0),
    pre_retry_hook: Callable[[], None] = None,
) -> tuple[T, int]:
    """
    Execute an operation with exponential backoff retry.

    Args:
        operation: Callable to execute
        operation_name: Name for logging
        max_attempts: Maximum retry attempts
        initial_backoff: Initial sleep time in seconds
        max_backoff: Maximum sleep time cap
        backoff_factor: Multiplier for each retry
        jitter_range: (min, max) random jitter to add
        pre_retry_hook: Optional callable to run before each retry
                        (e.g., re-sign items, refresh tokens)

    Returns:
        Tuple of (result, attempt_count)

    Raises:
        RetryableError: If all retries exhausted on retryable error
        PermanentError: If a permanent error is encountered
    """
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            # Add jitter on first attempt to desynchronize fan-out
            if attempt == 1:
                time.sleep(random.uniform(0.005, 0.2))

            result = operation()

            if attempt > 1:
                logger.info(f"{operation_name} succeeded on attempt {attempt}")

            return result, attempt

        except Exception as e:
            last_exception = e

            if not _is_retryable_exception(e):
                logger.error(f"{operation_name} permanent error: {e}")
                raise PermanentError(
                    f"{operation_name} failed with permanent error: {e}",
                    original_exception=e
                )

            if attempt == max_attempts:
                logger.error(
                    f"{operation_name} failed after {max_attempts} attempts: {e}"
                )
                raise RetryableError(
                    f"{operation_name} failed after {max_attempts} retries: {e}",
                    original_exception=e,
                    attempt=attempt,
                    operation=operation_name
                )

            # Calculate backoff with exponential growth and jitter
            sleep_s = min(
                initial_backoff * (backoff_factor ** (attempt - 1)),
                max_backoff
            )
            sleep_s += random.uniform(*jitter_range)

            logger.warning(
                f"{operation_name} error on attempt {attempt}/{max_attempts}: {e}; "
                f"retrying in {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)

            # Run pre-retry hook (e.g., re-sign items)
            if pre_retry_hook is not None:
                try:
                    pre_retry_hook()
                except Exception as hook_err:
                    logger.warning(f"Pre-retry hook failed: {hook_err}")

    # Yeesh should not get this far, but safety fallback anyway
    raise RetryableError(
        f"{operation_name} failed unexpectedly",
        original_exception=last_exception,
        attempt=max_attempts,
        operation=operation_name
    )


# -------------------- Constants --------------------
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
RTC_COLLECTION = "sentinel-1-rtc"
ASSETS_PREF = ["vv", "vh"]  # desired order



# -------------------- Planetary Computer SAS helpers --------------------
def _normalize_sas_token(token: str) -> str:
    """Normalize SAS token to a bare querystring (no leading '?')."""
    token = (token or "").strip()
    if token.startswith("?"):
        token = token[1:]
    return token

def append_sas_to_href(href: str, sas_token: str) -> str:
    """Append/merge a SAS token querystring onto a URL."""
    sas_token = _normalize_sas_token(sas_token)
    if not sas_token:
        return href
    u = urlparse(href)
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    q.update(dict(parse_qsl(sas_token, keep_blank_values=True)))
    return urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(q), u.fragment))

def apply_sas_to_item_assets(items, sas_token: str, asset_keys: list[str] | None = None):
    """Mutate PySTAC Items in-place to append SAS tokens to selected asset hrefs."""
    sas_token = _normalize_sas_token(sas_token)
    if not sas_token:
        return items
    for it in items:
        # PySTAC Item: .assets is a dict[str, Asset]
        if not getattr(it, "assets", None):
            continue
        keys = asset_keys or list(it.assets.keys())
        for k in keys:
            a = it.assets.get(k)
            if a and getattr(a, "href", None):
                a.href = append_sas_to_href(a.href, sas_token)
    return items

# -------------------- Small utils --------------------
def _elapsed_ms(t_start: float) -> float:
    return (time.perf_counter() - t_start) * 1000.0

def make_bbox(initial_bbx: list, expansion: int = 10) -> list:
    """Expand a point bbox by ~degrees (same style as your code)."""
    multiplier = 1 / 360
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
    """Compute fraction of tile area covered by the item's footprint."""
    try:
        tile_poly = box(*tile_bounds)
        item_poly = shape(item.geometry)
        inter = tile_poly.intersection(item_poly)
        if tile_poly.is_empty or tile_poly.area == 0:
            return 0.0
        return float(inter.area / tile_poly.area)
    except Exception as e:
        logger.warning(f"coverage_fraction failed for {getattr(item, 'id', 'unknown')}: {e}")
        return 0.0


# -------------------- Solar Day Helpers --------------------
def _get_solar_day(item) -> str | None:
    """Extract date string (YYYY-MM-DD) from item datetime."""
    dt = item.datetime if item.datetime else None
    if dt is None:
        dt_str = item.properties.get("datetime")
        if dt_str:
            return dt_str[:10]  # "2024-01-15T..." -> "2024-01-15"
        return None
    return dt.strftime("%Y-%m-%d")


def _group_items_by_day(items: list) -> dict[str, list]:
    """Group STAC items by solar day."""
    groups = {}
    for item in items:
        day = _get_solar_day(item)
        if day:
            if day not in groups:
                groups[day] = []
            groups[day].append(item)
    return groups


def _daily_coverage_fraction(items: list, tile_bounds: tuple) -> float:
    """Calculate combined coverage fraction for multiple items (same day).

    Unions all item footprints then intersects with tile.
    """
    if not items:
        return 0.0
    try:
        tile_poly = box(*tile_bounds)

        # Union all item footprints
        combined_footprint = None
        for item in items:
            item_poly = shape(item.geometry)
            if combined_footprint is None:
                combined_footprint = item_poly
            else:
                combined_footprint = combined_footprint.union(item_poly)

        if combined_footprint is None:
            return 0.0

        inter = tile_poly.intersection(combined_footprint)
        if tile_poly.is_empty or tile_poly.area == 0:
            return 0.0
        return float(inter.area / tile_poly.area)
    except Exception as e:
        logger.warning(f"_daily_coverage_fraction failed: {e}")
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
    vals = arr.astype(np.float32)
    total = int(vals.size)
    mask = vals > 0
    valid = vals[mask]
    if valid.size == 0:
        return {"min": 0, "max": 0, "mean": 0.0, "std": 0.0,
                "p5": 0.0, "p50": 0.0, "p95": 0.0,
                "valid_ratio": 0.0, "count": total, "valid_count": 0}
    p5, p50, p95 = np.percentile(valid, [5, 50, 95])
    return {"min": int(valid.min()), "max": int(valid.max()),
            "mean": float(valid.mean()), "std": float(valid.std()),
            "p5": float(p5), "p50": float(p50), "p95": float(p95),
            "valid_ratio": float(valid.size / total) if total > 0 else 0.0,
            "count": total, "valid_count": int(valid.size)}

def _to_numpy(x) -> np.ndarray:
    """Robust conversion to NumPy array across xarray/dask versions."""
    try:
        return x.to_numpy()
    except AttributeError:
        return np.asarray(x)


def load_quarter_items_odc(items, bbox, resolution_m: int, bands: list) -> np.ndarray | None:
    """
    Load S1 RTC items using odc.stac at specified resolution.

    Args:
        items: List of signed STAC items
        bbox: Bounding box as [minx, miny, maxx, maxy]
        resolution_m: Target resolution in meters (e.g., 40)
        bands: List of band names (e.g., ["vv", "vh"])

    Returns:
        Array of shape (T, bands, H, W) as uint16, or None if no items.
        T = number of unique solar days (same-day scenes are mosaicked together).
    """
    if not items:
        return None

    ds = stac_load(
        items,
        bands=bands,
        bbox=bbox,
        resolution=resolution_m,
        groupby="solar_day",  # Mosaic same-day scenes together
        resampling="bilinear",
        dtype="float32",
        chunks={}
    )

    # Stack bands: (T, H, W) per band -> (T, bands, H, W)
    arrays = []
    for band in bands:
        if band in ds:
            arr = _to_numpy(ds[band])  # (T, H, W)
            arrays.append(arr)
        else:
            logger.warning(f"Band '{band}' not found in dataset")

    if not arrays:
        return None

    stacked = np.stack(arrays, axis=1)  # (T, bands, H, W)

    # Scale [0,1] -> uint16 [0,65535], handling nodata
    stacked = np.where(np.isfinite(stacked) & (stacked > 0), stacked, 0.0)
    stacked = np.clip(stacked, 0.0, 1.0) * 65535.0
    return stacked.astype(np.uint16)


def _orbit_query(orbit_direction: str) -> dict:
    """Planetary Computer uses sat:orbit_state (ASCENDING/DESCENDING) commonly."""
    od = (orbit_direction or "BOTH").upper()
    if od == "ASCENDING":
        return {"sat:orbit_state": {"eq": "ascending"}}
    if od == "DESCENDING":
        return {"sat:orbit_state": {"eq": "descending"}}
    return {}


# -------------------- Orbit Direction Helpers --------------------
def _group_by_orbit(items: list) -> dict[str, list]:
    """Group STAC items by orbit direction."""
    groups = {"ascending": [], "descending": []}
    for item in items:
        orbit = item.properties.get("sat:orbit_state", "").lower()
        if orbit in groups:
            groups[orbit].append(item)
    return groups


def _count_qualifying_items(
    items: list,
    tile_bounds: tuple,
    coverage_threshold: float,
    k_scenes: int,
    year: int,
) -> dict:
    """Count qualifying days per quarter for a set of items.

    Groups items by solar day and counts days where combined coverage
    (union of footprints) meets the threshold.

    Returns dict with day counts and total qualifying days.
    """
    quarters = _quarter_windows(year)
    total = 0
    per_quarter = {}

    for q_name, (start, end) in quarters.items():
        start_dt = np.datetime64(start)
        end_dt = np.datetime64(end)

        # Find items in this quarter's time window
        quarter_items = []
        for item in items:
            dt = item.datetime.isoformat() if item.datetime else item.properties.get("datetime")
            if not dt:
                continue
            item_dt = np.datetime64(dt)
            if start_dt <= item_dt <= end_dt:
                quarter_items.append(item)

        # Group by solar day
        daily_groups = _group_items_by_day(quarter_items)

        # Count days meeting threshold (using combined daily coverage)
        qualifying_days = 0
        for day, day_items in daily_groups.items():
            if _daily_coverage_fraction(day_items, tile_bounds) >= coverage_threshold:
                qualifying_days += 1

        # Count top-k (or all if fewer)
        count = min(qualifying_days, k_scenes)
        per_quarter[q_name] = count
        total += count

    return {"total": total, "per_quarter": per_quarter, "raw_count": len(items)}


# -------------------- Debug Info Helpers --------------------
def _build_stac_debug_url(year: int, bbox: list, orbit_direction: str = None) -> str:
    """Build a STAC API URL for debugging (can be opened in browser/curl)."""
    base = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    params = {
        "collections": RTC_COLLECTION,
        "datetime": f"{year}-01-01/{year}-12-31",
        "bbox": ",".join(str(b) for b in bbox),
        "limit": "100",
    }
    if orbit_direction and orbit_direction.upper() not in ["BOTH", "AUTO"]:
        params["query"] = json.dumps({"sat:orbit_state": {"eq": orbit_direction.lower()}})

    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{base}?{query_string}"


def _build_stac_debug_info(
    year: int,
    bbox: list,
    orbit_direction: str,
    asc_count: int = 0,
    desc_count: int = 0,
) -> dict:
    """Build debug info for error messages."""
    return {
        "stac_url": _build_stac_debug_url(year, bbox, orbit_direction),
        "year": year,
        "bbox": bbox,
        "orbit_direction": orbit_direction,
        "items_found": {"ascending": asc_count, "descending": desc_count},
    }


def _quarter_windows(year: int):
    # Same quarter “center-ish” windows used, dont @ me.
    return {
        "Q1": (f"{year}-01-15", f"{year}-03-15"),
        "Q2": (f"{year}-04-15", f"{year}-06-15"),
        "Q3": (f"{year}-07-15", f"{year}-09-15"),
        "Q4": (f"{year}-10-15", f"{year}-12-15"),
    }

def get_quarterly_scenes_by_coverage(items: list, year: int, tile_bounds: tuple,
                                     coverage_threshold: float = 0.95,
                                     k_scenes: int = 3) -> tuple:
    """Select top-k days per quarter based on combined daily coverage.

    Groups items by solar day and calculates combined coverage (union of footprints)
    for each day. This allows edge-of-swath scenes to be combined for better coverage.
    """
    quarters = _quarter_windows(year)
    selected = {}
    quarter_meta = []

    for q_name, (start, end) in quarters.items():
        start_dt = np.datetime64(start)
        end_dt = np.datetime64(end)

        # Filter items to loop's quarter
        quarter_items = []
        for item in items:
            dt = item.datetime.isoformat() if item.datetime else item.properties.get("datetime")
            if not dt:
                continue
            item_dt = np.datetime64(dt)
            if start_dt <= item_dt <= end_dt:
                quarter_items.append(item)

        if not quarter_items:
            logger.warning(f"{q_name}: No scenes found")
            selected[q_name] = []
            quarter_meta.append({
                "quarter": q_name, "candidate_days": 0, "candidate_count": 0,
                "selected_days": [], "selected_ids": [],
                "selected_datetimes": [], "selected_orbit": None,
                "daily_coverage_fractions": [], "k_selected": 0
            })
            continue

        # Group by solar day
        daily_groups = _group_items_by_day(quarter_items)

        # Score each day by combined coverage (union of footprints)
        scored_days = []
        for day, day_items in daily_groups.items():
            combined_coverage = _daily_coverage_fraction(day_items, tile_bounds)
            scored_days.append((combined_coverage, day, day_items))

        scored_days.sort(key=lambda t: t[0], reverse=True)

        # Select top-k days meeting threshold
        top_k_days = []
        for coverage, day, day_items in scored_days[:k_scenes]:
            if coverage >= coverage_threshold or len(top_k_days) == 0:
                top_k_days.append((coverage, day, day_items))
            else:
                break
        if not top_k_days:
            top_k_days = [scored_days[0]]

        # Flatten: collect all items from selected days
        selected_items = []
        coverages = []
        selected_days_list = []
        for coverage, day, day_items in top_k_days:
            selected_items.extend(day_items)
            coverages.append(coverage)
            selected_days_list.append(day)

        selected[q_name] = selected_items

        quarter_meta.append({
            "quarter": q_name,
            "candidate_days": len(daily_groups),
            "candidate_count": len(quarter_items),
            "selected_days": selected_days_list,
            "selected_ids": [it.id for it in selected_items],
            "selected_datetimes": [_get_solar_day(it) for it in selected_items],
            "selected_orbit": selected_items[0].properties.get("sat:orbit_state") if selected_items else None,
            "daily_coverage_fractions": coverages,
            "k_selected": len(top_k_days),
        })

        logger.info(f"{q_name}: Selected {len(top_k_days)} day(s) with {len(selected_items)} scene(s), "
                   f"daily coverages={[f'{c:.3f}' for c in coverages]}")

    return selected, quarter_meta

def composite_quarter_scenes(scene_arrays: list, method: str = "median") -> np.ndarray:
    """
    scene_arrays: list of (bands,H,W) uint16 arrays
    returns: (bands,H,W) uint16
    """
    if not scene_arrays:
        raise ValueError("No scenes to composite")
    if len(scene_arrays) == 1:
        return scene_arrays[0]
    stacked = np.stack(scene_arrays, axis=0).astype(np.float32)  # (k,b,h,w)
    stacked = np.where(stacked == 0, np.nan, stacked)
    with np.errstate(all="ignore"):
        if method == "median":
            out = np.nanmedian(stacked, axis=0)
        elif method == "mean":
            out = np.nanmean(stacked, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
    out = np.where(np.isnan(out), 0, out)
    return out.astype(np.uint16)

def _detect_vv_vh_assets(item) -> list:
    """Return asset keys in preferred order (vv,vh) if present; else best-effort detection."""
    keys = list(item.assets.keys())
    lower = {k.lower(): k for k in keys}

    found = []
    for want in ASSETS_PREF:
        if want in item.assets:
            found.append(want)
        elif want in lower:
            found.append(lower[want])

    # fallback: substring match
    if len(found) < 2:
        for k in keys:
            kl = k.lower()
            if "vv" in kl and all("vv" not in f.lower() for f in found):
                found.append(k)
            if "vh" in kl and all("vh" not in f.lower() for f in found):
                found.append(k)

    # keep order vv then vh if both exist
    def _rank(k):
        kl = k.lower()
        return 0 if "vv" in kl else (1 if "vh" in kl else 2)
    found = sorted(set(found), key=_rank)

    # keep only first two (vv,vh) if we got extras
    if len(found) > 2:
        found = found[:2]
    return found

# def read_rtc_band_to_tile_uint16(href: str, bounds_lonlat: tuple, target_crs: str = "EPSG:4326") -> np.ndarray:
#     """
#     Read RTC COG over the tile bounds, warped to EPSG:4326, return uint16 scaled.
#     Scaling: clip [0,1] -> [0,65535], matching your OPTION A style.
#     """
#     # Avoid GDAL reading directory listings for cloud paths
#     env = rio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")

#     with env:
#         with rio.open(href) as src:
#             with WarpedVRT(
#                 src,
#                 crs=target_crs,
#                 resampling=Resampling.bilinear,
#                 dst_nodata=0,
#             ) as vrt:
#                 win = from_bounds(*bounds_lonlat, transform=vrt.transform)
#                 win = win.round_offsets().round_lengths()
#                 arr = vrt.read(1, window=win).astype(np.float32)

#     # Replace nodata/negatives with 0 (RTC can have 0 nodata)
#     arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)

#     # Your scaling OPTION A
#     scaled = np.clip(arr, 0.0, 1.0) * 65535.0
#     return scaled.astype(np.uint16)


# -------------------- STAC Search with Retry --------------------
def _search_stac_items(
    client: Client,
    year: int,
    bbox_geojson: dict,
    orbit_query: dict,
    max_attempts: int = 5,
    initial_backoff: float = 0.5,
) -> tuple[list, int]:
    """Search STAC with retry logic.

    Returns:
        Tuple of (items_list, attempt_count)
    """
    def _do_search():
        search = client.search(
            collections=[RTC_COLLECTION],
            datetime=f"{year}-01-01/{year}-12-31",
            intersects=bbox_geojson,
            query=orbit_query,
            limit=500,
        )
        return list(search.items())

    return retry_with_backoff(
        operation=_do_search,
        operation_name="STAC_search",
        max_attempts=max_attempts,
        initial_backoff=initial_backoff,
    )


# -------------------- Quarter Result Dataclass --------------------
@dataclass
class QuarterResult:
    """Result of processing a single quarter."""
    quarter: str
    status: str  # 'success', 'failed', 'empty'
    data: np.ndarray | None
    items_loaded: int
    retry_count: int
    error_message: str | None = None
    error_type: str | None = None  # 'retryable', 'permanent'


# -------------------- Quarter Loading with Retry --------------------
def _sign_items_with_retry(
    items: list,
    sas_token: str,
    bands: list,
    max_attempts: int = 3,
) -> list:
    """Sign items with retry, preferring pre-fetched SAS token."""
    if sas_token:
        apply_sas_to_item_assets(items, sas_token, asset_keys=bands)
        return items

    # Fallback: sign each item individually with retry
    signed_items = []
    for item in items:
        def _sign_single(it=item):
            return pc.sign(it)

        signed_item, _ = retry_with_backoff(
            operation=_sign_single,
            operation_name=f"PC_sign_{item.id}",
            max_attempts=max_attempts,
            initial_backoff=0.3,
            jitter_range=(0.05, 0.2),
        )
        signed_items.append(signed_item)

    return signed_items


def _load_quarter_with_retry(
    quarter_name: str,
    items: list,
    bbox: list,
    resolution_m: int,
    bands: list,
    sas_token: str,
    max_attempts: int = 3,
    fallback_shape: tuple = (512, 512),
) -> QuarterResult:
    """
    Load quarter data with retry, re-signing items before each retry.
    """
    if not items:
        return QuarterResult(
            quarter=quarter_name,
            status='empty',
            data=np.zeros((len(bands), *fallback_shape), dtype=np.uint16),
            items_loaded=0,
            retry_count=0,
        )

    # Create working copy for re-signing
    working_items = items.copy()

    def _re_sign_items():
        nonlocal working_items
        if sas_token:
            apply_sas_to_item_assets(working_items, sas_token, asset_keys=bands)
        else:
            working_items = [pc.sign(it) for it in items]

    def _do_load():
        return load_quarter_items_odc(working_items, bbox, resolution_m, bands)

    try:
        loaded, attempt_count = retry_with_backoff(
            operation=_do_load,
            operation_name=f"load_{quarter_name}",
            max_attempts=max_attempts,
            initial_backoff=1.0,
            pre_retry_hook=_re_sign_items,
        )

        if loaded is None or loaded.size == 0:
            return QuarterResult(
                quarter=quarter_name,
                status='empty',
                data=np.zeros((len(bands), *fallback_shape), dtype=np.uint16),
                items_loaded=0,
                retry_count=attempt_count - 1,
            )

        # Composite if multiple scenes
        if loaded.shape[0] > 1:
            loaded_f = loaded.astype(np.float32)
            loaded_f = np.where(loaded_f == 0, np.nan, loaded_f)
            with np.errstate(all="ignore"):
                quarter_data = np.nanmedian(loaded_f, axis=0)
            quarter_data = np.where(np.isnan(quarter_data), 0, quarter_data)
            quarter_data = quarter_data.astype(np.uint16)
        else:
            quarter_data = loaded[0]

        logger.debug(f"  {quarter_name} composite shape: {quarter_data.shape}, range: {quarter_data.min()}..{quarter_data.max()}")

        return QuarterResult(
            quarter=quarter_name,
            status='success',
            data=quarter_data,
            items_loaded=loaded.shape[0],
            retry_count=attempt_count - 1,
        )

    except RetryableError as e:
        logger.warning(f"{quarter_name} failed after retries: {e}")
        return QuarterResult(
            quarter=quarter_name,
            status='failed',
            data=np.zeros((len(bands), *fallback_shape), dtype=np.uint16),
            items_loaded=0,
            retry_count=e.attempt,
            error_message=str(e),
            error_type='retryable',
        )
    except PermanentError as e:
        logger.error(f"{quarter_name} permanent failure: {e}")
        return QuarterResult(
            quarter=quarter_name,
            status='failed',
            data=np.zeros((len(bands), *fallback_shape), dtype=np.uint16),
            items_loaded=0,
            retry_count=0,
            error_message=str(e),
            error_type='permanent',
        )


# -------------------- Main --------------------
def main() -> dict | None:
    ap = argparse.ArgumentParser(description="Planetary Computer S1 RTC quarterly composites -> HKL")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--X_tile", type=int, required=True)
    ap.add_argument("--Y_tile", type=int, required=True)
    ap.add_argument("--dest", type=str, required=True)
    ap.add_argument("--expansion", type=int, default=300)
    ap.add_argument("--coverage-threshold", type=float, default=0.95)
    ap.add_argument("--orbit-direction", type=str, choices=["AUTO", "ASCENDING", "DESCENDING", "BOTH"], default="AUTO",
                    help="Orbit selection: AUTO picks direction with most qualifying data (default)")
    ap.add_argument("--k-scenes", type=int, default=3)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--run-metadata", action="store_true")
    ap.add_argument("--sas-token", type=str, default=os.getenv("PC_SAS_TOKEN", ""),
                    help="Optional Planetary Computer SAS token (querystring) to append to asset hrefs. "
                         "If provided, avoids per-item signing calls from the Lambda.")
    ap.add_argument("--resolution", type=int, default=40,
                    help="Target resolution in meters (default: 40)")
    args = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.debug else "INFO")

    t_all = time.perf_counter()

    # Output store (local or s3)
    if args.dest.startswith("s3://"):
        store = from_url(args.dest, region="us-east-1") #us-west-2 if dl s3, us-east-1 if gri #TODO: move to  storage region arg but needs to pass from orchestrator script
    else:
        os.makedirs(args.dest, exist_ok=True)
        store = LocalStore(prefix=args.dest)
    base_key = f"{args.year}/raw/{args.X_tile}/{args.Y_tile}/raw"
    s1_dir_key = f"{base_key}/s1"
    misc_key = f"{base_key}/misc"
    fn_s1_key = f"{s1_dir_key}/{args.X_tile}X{args.Y_tile}Y.hkl"
    fn_s1_dates_key = f"{misc_key}/s1_dates_{args.X_tile}X{args.Y_tile}Y.hkl"

    configure_rio(cloud_defaults=True)

    # Build bbox
    initial_bbx = [args.lon, args.lat, args.lon, args.lat]
    bbx = make_bbox(initial_bbx, expansion=args.expansion / 30)
    tile_bounds = featureBounds(bbox2geojson(bbx))  # (minx,miny,maxx,maxy) in lon/lat
    logger.info(f"Tile: {args.X_tile}X{args.Y_tile}Y, {args.year}, {tile_bounds}")
    # STAC query (Planetary Computer) with retry
    client = Client.open(PC_STAC)

    # Track orbit selection info for return metadata
    orbit_selection_info = {
        "orbit_selected": None,
        "orbit_stats": {"ascending": 0, "descending": 0},
        "selection_method": args.orbit_direction,
    }

    if args.orbit_direction.upper() == "AUTO":
        # Query without orbit filter to get all items
        stac_query = {}
        items, stac_attempts = _search_stac_items(
            client, args.year, bbox2geojson(bbx), stac_query
        )
        if stac_attempts > 1:
            logger.info(f"STAC search succeeded after {stac_attempts} attempts")

        if not items:
            debug_info = _build_stac_debug_info(args.year, bbx, args.orbit_direction)
            raise PermanentError(
                f"No Sentinel-1 RTC items found for query. "
                f"STAC URL: {debug_info['stac_url']}"
            )

        logger.info(f"Found {len(items)} PC RTC scenes for {args.year} (all orbits)")

        # Group by orbit direction
        orbit_groups = _group_by_orbit(items)

        # Count qualifying items per direction
        asc_stats = _count_qualifying_items(
            orbit_groups["ascending"], tile_bounds,
            args.coverage_threshold, args.k_scenes, args.year
        )
        desc_stats = _count_qualifying_items(
            orbit_groups["descending"], tile_bounds,
            args.coverage_threshold, args.k_scenes, args.year
        )

        orbit_selection_info["orbit_stats"] = {
            "ascending": asc_stats["total"],
            "descending": desc_stats["total"],
        }

        # Pick direction with more qualifying items
        # If qualifying counts are equal, use raw item count as tie-breaker
        # Prefer ascending only on complete ties (same qualifying AND raw counts)
        asc_key = (asc_stats["total"], asc_stats["raw_count"])
        desc_key = (desc_stats["total"], desc_stats["raw_count"])

        if asc_key >= desc_key:
            selected_orbit = "ascending"
            items = orbit_groups["ascending"]
            selected_stats = asc_stats
        else:
            selected_orbit = "descending"
            items = orbit_groups["descending"]
            selected_stats = desc_stats

        orbit_selection_info["orbit_selected"] = selected_orbit.upper()

        logger.info(
            f"Selected {selected_orbit.upper()} orbit: {selected_stats['total']} qualifying items "
            f"(ASC: {asc_stats['total']}/{asc_stats['raw_count']} qual/raw, "
            f"DESC: {desc_stats['total']}/{desc_stats['raw_count']} qual/raw)"
        )

        if not items:
            debug_info = _build_stac_debug_info(
                args.year, bbx, args.orbit_direction,
                asc_count=len(orbit_groups["ascending"]),
                desc_count=len(orbit_groups["descending"])
            )
            raise PermanentError(
                f"No items in selected orbit direction ({selected_orbit}). "
                f"STAC URL: {debug_info['stac_url']}"
            )
    else:
        # Use specified orbit direction (ASCENDING, DESCENDING, or BOTH)
        stac_query = _orbit_query(args.orbit_direction)
        items, stac_attempts = _search_stac_items(
            client, args.year, bbox2geojson(bbx), stac_query
        )
        if stac_attempts > 1:
            logger.info(f"STAC search succeeded after {stac_attempts} attempts")

        if not items:
            debug_info = _build_stac_debug_info(args.year, bbx, args.orbit_direction)
            raise PermanentError(
                f"No Sentinel-1 RTC items found for query. "
                f"STAC URL: {debug_info['stac_url']}"
            )

        orbit_selection_info["orbit_selected"] = args.orbit_direction.upper()
        logger.info(f"Found {len(items)} PC RTC scenes for {args.year} ({args.orbit_direction})")

    # NOTE: Do not sign all items up front; we only sign the small selected subset per quarter.

    # Select top-K per quarter by coverage
    quarterly_items, quarter_meta = get_quarterly_scenes_by_coverage(
        items, args.year, tile_bounds, args.coverage_threshold, args.k_scenes
    )

    # Determine bands/asset keys from available items (prefer vv,vh)
    bands = None
    for q, lst in quarterly_items.items():
        if lst:
            bands = _detect_vv_vh_assets(lst[0])
            break
    if not bands or len(bands) == 0:
        bands = ["vv"]
        logger.warning("Could not detect vv/vh assets; falling back to vv only")
    else:
        # ensure vv then vh ordering
        bands = sorted(bands, key=lambda k: 0 if "vv" in k.lower() else (1 if "vh" in k.lower() else 2))
    logger.info(f"Using asset keys: {bands}")

    quarter_results: dict[str, QuarterResult] = {}
    quarter_band_stats = {q: {} for q in ["Q1", "Q2", "Q3", "Q4"]}

    # Process each quarter using odc.stac with retry
    for q_name in ["Q1", "Q2", "Q3", "Q4"]:
        item_list = quarterly_items.get(q_name, [])

        if item_list:
            logger.info(f"Processing {q_name}: {len(item_list)} scene(s) at {args.resolution}m resolution")
            for item in item_list:
                logger.debug(f"  Item: {item.id}")

            # Sign items before loading (with retry for fallback signing)
            item_list = _sign_items_with_retry(item_list, args.sas_token, bands)

        # Load with retry
        result = _load_quarter_with_retry(
            quarter_name=q_name,
            items=item_list,
            bbox=bbx,
            resolution_m=args.resolution,
            bands=bands,
            sas_token=args.sas_token,
        )
        quarter_results[q_name] = result

        # Stats for successful quarters
        if result.status == 'success' and result.data is not None:
            try:
                for bi, bname in enumerate(bands):
                    quarter_band_stats[q_name][bname] = compute_band_stats(result.data[bi])
            except Exception as e:
                logger.warning(f"Failed computing band stats for {q_name}: {e}")

    # Collect only successful quarter data (exclude failed quarters from output)
    successful_quarters = []
    successful_quarter_names = []
    quarter_days_map = {"Q1": 45, "Q2": 135, "Q3": 225, "Q4": 315}

    for q_name in ["Q1", "Q2", "Q3", "Q4"]:
        result = quarter_results[q_name]
        if result.status == 'success':
            successful_quarters.append(result.data)
            successful_quarter_names.append(q_name)
        elif result.status == 'empty':
            # Empty quarters (no items found) are still included with zeros
            successful_quarters.append(result.data)
            successful_quarter_names.append(q_name)
        # Failed quarters are excluded from output

    # Determine overall status
    succeeded_list = [q for q, r in quarter_results.items() if r.status in ('success', 'empty')]
    failed_list = [q for q, r in quarter_results.items() if r.status == 'failed']

    if not successful_quarters:
        # All quarters failed - don't save, raise error
        error_details = "; ".join(
            f"{q}: {quarter_results[q].error_message}"
            for q in failed_list
        )
        raise PermanentError(f"All quarters failed: {error_details}")

    if failed_list:
        logger.warning(f"Partial success: {len(succeeded_list)} quarters succeeded, {len(failed_list)} failed ({failed_list})")

    # Ensure all quarters same shape
    max_bands = max(a.shape[0] for a in successful_quarters)
    max_h = max(a.shape[1] for a in successful_quarters)
    max_w = max(a.shape[2] for a in successful_quarters)

    def _pad(arr):
        pad_b = max_bands - arr.shape[0]
        pad_h = max_h - arr.shape[1]
        pad_w = max_w - arr.shape[2]
        if pad_b > 0:
            arr = np.pad(arr, ((0, pad_b), (0, 0), (0, 0)), mode="edge")
        if pad_h > 0 or pad_w > 0:
            mode = "reflect" if (arr.shape[1] > 1 and arr.shape[2] > 1) else "edge"
            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode=mode)
        return arr

    successful_quarters = [_pad(a) for a in successful_quarters]

    # Stack successful quarters, repeat each 3 times, transpose
    all_quarters = np.stack(successful_quarters, axis=0)       # (N,B,H,W) where N <= 4
    repeated = np.repeat(all_quarters, 3, axis=0)              # (N*3,B,H,W)
    final_array = np.transpose(repeated, (0, 2, 3, 1))         # (N*3,H,W,B)

    # Dates for successful quarters only
    s1_dates = np.array(
        [quarter_days_map[q] for q in successful_quarter_names for _ in range(3)],
        dtype=np.int64
    )

    # Write outputs
    t_write = time.perf_counter()
    obstore_put_hkl(store, fn_s1_key, final_array)
    obstore_put_hkl(store, fn_s1_dates_key, s1_dates)

    full_s1_path = f"{args.dest.rstrip('/')}/{fn_s1_key}"
    full_dates_path = f"{args.dest.rstrip('/')}/{fn_s1_dates_key}"

    logger.success(f"S1 RTC quarterly composites saved: {full_s1_path}")
    logger.success(f"Shape: {final_array.shape}, dtype: {final_array.dtype}")
    logger.success(f"Range: {int(final_array.min())} to {int(final_array.max())}")
    logger.success(f"S1 quarterly dates saved: {full_dates_path}")
    logger.info(f"Write {_elapsed_ms(t_write):.0f} ms; total {_elapsed_ms(t_all):.0f} ms")

    # Optional metadata sidecar
    if args.run_metadata:
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            meta_key = f"{misc_key}/s1_meta_{args.X_tile}X{args.Y_tile}Y_{ts}.txt"
            lines = []
            lines.append("Planetary Computer Sentinel-1 RTC quarterly processing metadata")
            lines.append(f"timestamp: {ts}")
            lines.append(f"year: {args.year}")
            lines.append(f"lon: {args.lon}")
            lines.append(f"lat: {args.lat}")
            lines.append(f"X_tile: {args.X_tile}")
            lines.append(f"Y_tile: {args.Y_tile}")
            lines.append(f"bbox: {bbx}")
            lines.append(f"bounds: {tile_bounds}")
            lines.append(f"resolution_m: {args.resolution}")
            lines.append(f"items_found: {len(items)}")
            lines.append(f"orbit_direction: {args.orbit_direction}")
            lines.append(f"orbit_selected: {orbit_selection_info['orbit_selected']}")
            lines.append(f"orbit_stats: {orbit_selection_info['orbit_stats']}")
            lines.append(f"coverage_threshold: {args.coverage_threshold}")
            lines.append(f"k_scenes: {args.k_scenes}")
            lines.append(f"assets: {bands}")
            lines.append("quarters:")
            for qm in quarter_meta:
                lines.append(f"  - quarter: {qm['quarter']}")
                lines.append(f"    k_selected_days: {qm['k_selected']}")
                lines.append(f"    selected_days: {qm.get('selected_days', [])}")
                lines.append(f"    selected_ids: {qm['selected_ids']}")
                lines.append(f"    selected_datetimes: {qm['selected_datetimes']}")
                lines.append(f"    orbit: {qm['selected_orbit']}")
                lines.append(f"    daily_coverage_fractions: {qm.get('daily_coverage_fractions', qm.get('coverage_fractions', []))}")
                lines.append(f"    candidate_days: {qm.get('candidate_days', 0)}")
                lines.append(f"    candidate_scenes: {qm['candidate_count']}")
            lines.append(f"output_shape: {list(final_array.shape)}")
            lines.append(f"output_dtype: {str(final_array.dtype)}")
            lines.append(f"output_min: {int(final_array.min())}")
            lines.append(f"output_max: {int(final_array.max())}")
            # per-quarter stats
            lines.append("quarter_band_stats:")
            for qn in ["Q1","Q2","Q3","Q4"]:
                lines.append(f"  {qn}:")
                qb = quarter_band_stats.get(qn, {})
                for bname in bands:
                    st = qb.get(bname)
                    if not st:
                        continue
                    lines.append(f"    {bname}:")
                    lines.append(f"      min: {st['min']}")
                    lines.append(f"      max: {st['max']}")
                    lines.append(f"      mean: {st['mean']:.3f}")
                    lines.append(f"      std: {st['std']:.3f}")
                    lines.append(f"      p5: {st['p5']:.3f}")
                    lines.append(f"      p50: {st['p50']:.3f}")
                    lines.append(f"      p95: {st['p95']:.3f}")
                    lines.append(f"      valid_ratio: {st['valid_ratio']:.4f}")
                    lines.append(f"      count: {st['count']}")
                    lines.append(f"      valid_count: {st['valid_count']}")
            lines.append(f"s1_key: {fn_s1_key}")
            lines.append(f"dates_key: {fn_s1_dates_key}")
            lines.append(f"write_ms: {_elapsed_ms(t_write):.0f}")
            lines.append(f"total_ms: {_elapsed_ms(t_all):.0f}")

            obstore_put_text(store, meta_key, "\n".join(lines) + "\n")
            logger.success(f"S1 metadata sidecar saved: {args.dest.rstrip('/')}/{meta_key}")
        except Exception as e:
            logger.error(f"Failed to write metadata sidecar: {e}")

    # Return structured result for programmatic use
    return {
        "status": "partial" if failed_list else "success",
        "quarters_succeeded": succeeded_list,
        "quarters_failed": failed_list,
        "quarter_details": {
            q: {
                "status": r.status,
                "items_loaded": r.items_loaded,
                "retry_count": r.retry_count,
                "error_message": r.error_message,
                "error_type": r.error_type,
            }
            for q, r in quarter_results.items()
        },
        "total_retry_count": sum(r.retry_count for r in quarter_results.values()),
        "stac_search_attempts": stac_attempts,
        "output_shape": list(final_array.shape),
        "output_dtype": str(final_array.dtype),
        "elapsed_ms": _elapsed_ms(t_all),
        "orbit_selected": orbit_selection_info["orbit_selected"],
        "orbit_stats": orbit_selection_info["orbit_stats"],
    }


# --- programmatic entrypoint for Lithops parity ---
def run(
    year: int | str,
    lon: float,
    lat: float,
    X_tile: int | str,
    Y_tile: int | str,
    dest: str,
    expansion: int = 300,
    debug: bool = False,
    run_metadata: bool = False,
    orbit_direction: str = "AUTO",
    k_scenes: int = 3,
    coverage_threshold: float = 0.95,
    resolution: int = 40,
    sas_token: str = "",
) -> dict:
    """
    Programmatic entrypoint for Lithops.

    Always returns a dict. Never raises exceptions.
    Status will be 'success', 'partial', 'failed', or 'error'.
    """
    import sys
    t_start = time.perf_counter()

    # Base info included in all responses
    base_info = {
        "product": "s1_rtc_pc",
        "year": int(year),
        "lon": float(lon),
        "lat": float(lat),
        "X_tile": int(X_tile),
        "Y_tile": int(Y_tile),
        "dest": dest,
        "orbit_direction": orbit_direction,
        "k_scenes": int(k_scenes),
        "coverage_threshold": float(coverage_threshold),
        "resolution": int(resolution),
        "used_shared_sas_token": bool(sas_token),
    }

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
        "--coverage-threshold", str(coverage_threshold),
        "--resolution", str(resolution),
    ]
    if sas_token:
        argv += ["--sas-token", sas_token]
    if debug:
        argv.append("--debug")
    if run_metadata:
        argv.append("--run-metadata")

    old = sys.argv
    try:
        sys.argv = argv
        result = main()

        # Merge base info with main() result
        if result is None:
            result = {}

        return {
            **base_info,
            "status": result.get("status", "success"),
            "quarters_succeeded": result.get("quarters_succeeded", ["Q1", "Q2", "Q3", "Q4"]),
            "quarters_failed": result.get("quarters_failed", []),
            "quarter_details": result.get("quarter_details", {}),
            "total_retry_count": result.get("total_retry_count", 0),
            "stac_search_attempts": result.get("stac_search_attempts", 1),
            "output_shape": result.get("output_shape"),
            "elapsed_ms": result.get("elapsed_ms", _elapsed_ms(t_start)),
            "orbit_selected": result.get("orbit_selected"),
            "orbit_stats": result.get("orbit_stats"),
            "error_message": None,
            "error_type": None,
            "error_traceback": None,
            "debug_info": None,
        }

    except PermanentError as e:
        logger.error(f"Permanent error: {e}")
        # Build debug info for error context
        initial_bbx = [float(lon), float(lat), float(lon), float(lat)]
        bbx = make_bbox(initial_bbx, expansion=expansion / 30)
        debug_info = _build_stac_debug_info(int(year), bbx, orbit_direction)
        return {
            **base_info,
            "status": "error",
            "error_type": "permanent",
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
            "elapsed_ms": _elapsed_ms(t_start),
            "quarters_succeeded": [],
            "quarters_failed": [],
            "quarter_details": {},
            "total_retry_count": 0,
            "stac_search_attempts": 0,
            "output_shape": None,
            "orbit_selected": None,
            "orbit_stats": None,
            "debug_info": debug_info,
        }

    except RetryableError as e:
        logger.warning(f"Retryable error (exhausted retries): {e}")
        # Build debug info for error context
        initial_bbx = [float(lon), float(lat), float(lon), float(lat)]
        bbx = make_bbox(initial_bbx, expansion=expansion / 30)
        debug_info = _build_stac_debug_info(int(year), bbx, orbit_direction)
        return {
            **base_info,
            "status": "error",
            "error_type": "retryable",
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
            "elapsed_ms": _elapsed_ms(t_start),
            "quarters_succeeded": [],
            "quarters_failed": [],
            "quarter_details": {},
            "total_retry_count": getattr(e, 'attempt', 0),
            "stac_search_attempts": 0,
            "output_shape": None,
            "orbit_selected": None,
            "orbit_stats": None,
            "debug_info": debug_info,
        }

    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception(f"Unexpected error: {e}")
        is_retryable = _is_retryable_exception(e)
        # Build debug info for error context
        initial_bbx = [float(lon), float(lat), float(lon), float(lat)]
        bbx = make_bbox(initial_bbx, expansion=expansion / 30)
        debug_info = _build_stac_debug_info(int(year), bbx, orbit_direction)
        return {
            **base_info,
            "status": "error",
            "error_type": "retryable" if is_retryable else "permanent",
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
            "elapsed_ms": _elapsed_ms(t_start),
            "quarters_succeeded": [],
            "quarters_failed": [],
            "quarter_details": {},
            "total_retry_count": 0,
            "stac_search_attempts": 0,
            "output_shape": None,
            "orbit_selected": None,
            "orbit_stats": None,
            "debug_info": debug_info,
        }

    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
