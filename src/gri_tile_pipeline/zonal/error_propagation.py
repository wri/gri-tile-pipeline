"""4-source error combination in quadrature.

Ported from reference ``ttc_error_utils.py``.
Error sources:
  1. Shift error — 8-directional +-10m shift RMSE
  2. Small site error — for polygons < threshold area
  3. LULC error — ESA land cover classification uncertainty
  4. Subregion error — regional confidence intervals

Total = sqrt(shift^2 + small_site^2 + lulc^2 + subregion^2)
"""

from __future__ import annotations

import contextlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from gri_tile_pipeline.config import PipelineConfig


def _is_remote_path(path: str) -> bool:
    """Return True if path is a remote URI (s3://, gs://, https://)."""
    return path.startswith(("s3://", "gs://", "https://", "http://", "/vsis3/"))


@contextlib.contextmanager
def _rasterio_env(path: str):
    """Yield a rasterio.Env configured for efficient COG reads when *path* is remote.

    For local paths this is a no-op.
    """
    if _is_remote_path(path):
        import rasterio

        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
            GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
            GDAL_HTTP_MULTIPLEX="YES",
            GDAL_HTTP_VERSION=2,
        ):
            yield
    else:
        yield

# Data file paths (shipped with the package)
_DATA_DIR = Path(__file__).parent / "data"

# ESA WorldCover LULC class → category mapping (from reference config.yaml)
ESA_LULC_CONVERSIONS = {
    "urban": [190],
    "grassland": [130],
    "cropland": [10, 11, 12, 20],
    "forest": [50, 60, 61, 62, 70, 71, 80, 81, 82, 90, 160, 170],
    "mosaic vegetation": [30, 40, 100, 110],
    "shrub/scrub/otherland": [120, 121, 122, 140, 150, 151, 152, 153, 200, 201, 202, 0, 220],
}

# Small site error raw values (from reference config.yaml)
_SMALL_SITE_ERRORS = {
    (0.0, 10.0): 3.6386,    # 0 <= ttc <= 10
    (10.0, 40.0): 16.68,    # 10 < ttc <= 40
    (40.0, 100.0): 23.468,  # 40 < ttc <= 100
}


def _load_lulc_ci() -> pd.DataFrame:
    """Load LULC confidence interval table."""
    return pd.read_csv(_DATA_DIR / "lulc_ci.csv")


def _load_region_conf() -> pd.DataFrame:
    """Load regional confidence interval table."""
    return pd.read_csv(_DATA_DIR / "region_conf.csv")


def _load_subregions():
    """Load subregion polygons with confidence interval properties.

    Returns GeoDataFrame or None if file doesn't exist.
    """
    geojson_path = _DATA_DIR / "subregions_conf.geojson"
    if not geojson_path.exists():
        logger.warning(f"subregions_conf.geojson not found at {geojson_path}")
        return None
    import geopandas as gpd
    return gpd.read_file(geojson_path)


# -----------------------------------------------------------------------
# Individual error components
# -----------------------------------------------------------------------

def shift_error(
    mosaic_path: str,
    geometry,
    base_ttc: float,
    offset: float = 0.0001081081,  # ~10m in degrees (from reference config)
) -> float:
    """Compute shift error via 8-directional polygon displacement.

    Tests N, S, E, W and 4 diagonal shifts of *offset* degrees,
    recalculates mean TTC for each shifted polygon, and returns
    the RMSE of percent errors (matching reference formula).
    """
    import geopandas as gpd
    import rasterio
    from exactextract import exact_extract
    from shapely.affinity import translate

    if base_ttc == 0:
        return 0.0

    shifts = [
        (offset, 0), (-offset, 0),
        (0, offset), (0, -offset),
        (offset, offset), (offset, -offset),
        (-offset, offset), (-offset, -offset),
    ]

    percent_errors = []
    with rasterio.open(mosaic_path) as src:
        for dx, dy in shifts:
            shifted = translate(geometry, xoff=dx, yoff=dy)
            shifted_gdf = gpd.GeoDataFrame(geometry=[shifted], crs="EPSG:4326")
            tmp_file = tempfile.NamedTemporaryFile(
                suffix=".geojson", delete=False, mode="w"
            )
            tmp_file.write(shifted_gdf.to_json())
            tmp_file.close()
            try:
                result = exact_extract(
                    src,
                    tmp_file.name,
                    "mean(min_coverage_frac=0.05, coverage_weight=fraction)",
                )
                val = result[0]["properties"]["mean"]
                if val is not None and not np.isnan(val):
                    pct_err = (val - base_ttc) / base_ttc
                    percent_errors.append(pct_err ** 2)
            except Exception:
                pass
            finally:
                os.remove(tmp_file.name)

    if not percent_errors:
        return 0.0
    # RMS of percent errors over 8 shifts (matching reference)
    return float((sum(percent_errors) / 8) ** 0.5)


def small_site_error(
    area_ha: float,
    expected_ttc: float,
    threshold: float = 0.5,
) -> float:
    """Lookup small-site error for polygons below area threshold.

    Reference formula: raw_error_value / expected_ttc
    Brackets: {0-10: 3.6386, 10-40: 16.68, 40-100: 23.468}
    """
    if expected_ttc == 0:
        return 0.0
    if area_ha > threshold:
        return 0.0

    # First bracket: 0 <= ttc <= 10 (inclusive both ends)
    if 0.0 <= expected_ttc <= 10.0:
        return 3.6386 / expected_ttc
    # Second bracket: 10 < ttc <= 40
    elif expected_ttc <= 40.0:
        return 16.68 / expected_ttc
    # Third bracket: 40 < ttc <= 100
    elif expected_ttc <= 100.0:
        return 23.468 / expected_ttc
    else:
        return 0.0


def lulc_error(
    expected_ttc: float,
    lulc_category: str = "forest",
) -> Tuple[float, float]:
    """Lookup LULC confidence interval error.

    Reference formula:
        lower_error = (p_upper_95 - r_lower_95) / expected_ttc
        upper_error = (r_upper_95 - p_lower_95) / expected_ttc
    """
    if expected_ttc == 0:
        return 0.0, 0.0

    ci_df = _load_lulc_ci()
    ci_df["category"] = ci_df["category"].str.lower()
    match = ci_df[ci_df["category"] == lulc_category.lower()]
    if match.empty:
        logger.warning(f"LULC category '{lulc_category}' not found, using forest")
        match = ci_df[ci_df["category"] == "forest"]

    row = match.iloc[0]
    lower_error = (row["p_upper_95"] - row["r_lower_95"]) / expected_ttc
    upper_error = (row["r_upper_95"] - row["p_lower_95"]) / expected_ttc

    return float(lower_error), float(upper_error)


def subregion_error(
    geometry,
    expected_ttc: float,
    subregions_gdf=None,
) -> Tuple[Optional[str], float, float]:
    """Compute subregion error via spatial join with subregion boundaries.

    Reference formula:
        lower_error = (p_upper_95 - r_lower_95) / expected_ttc
        upper_error = (r_upper_95 - p_lower_95) / expected_ttc
    """
    if expected_ttc == 0:
        return None, 0.0, 0.0

    if subregions_gdf is None:
        subregions_gdf = _load_subregions()
    if subregions_gdf is None:
        logger.warning("No subregions data available, returning zero subregion error")
        return None, 0.0, 0.0

    centroid = geometry.centroid
    intersecting = subregions_gdf[subregions_gdf.intersects(centroid)]

    if intersecting.empty:
        logger.debug("No subregion intersection found")
        return None, 0.0, 0.0

    feat = intersecting.iloc[0]
    category = feat["category"]
    lower_error = (feat["p_upper_95"] - feat["r_lower_95"]) / expected_ttc
    upper_error = (feat["r_upper_95"] - feat["p_lower_95"]) / expected_ttc

    return category, float(lower_error), float(upper_error)


def combine_errors(
    expected_ttc: float,
    shift_err: float,
    small_site_err: float,
    lulc_lower: float,
    lulc_upper: float,
    subregion_lower: float,
    subregion_upper: float,
) -> Dict[str, float]:
    """Combine all 4 error sources in quadrature.

    Returns dict with ``error_plus``, ``error_minus``,
    ``plus_minus_average``.
    """
    shift_half = shift_err / 2
    small_half = small_site_err / 2

    lower = (lulc_lower ** 2 + subregion_lower ** 2 + shift_half ** 2 + small_half ** 2) ** 0.5
    upper = (lulc_upper ** 2 + subregion_upper ** 2 + shift_half ** 2 + small_half ** 2) ** 0.5

    minus = expected_ttc * lower
    plus = expected_ttc * upper
    avg = (minus + plus) / 2

    return {
        "error_plus": plus,
        "error_minus": minus,
        "plus_minus_average": avg,
    }


# -----------------------------------------------------------------------
# LULC raster integration
# -----------------------------------------------------------------------

def prep_lulc_data(polygons_gdf, lulc_raster_path: str, temp_dir: str) -> str:
    """Clip ESA WorldCover raster to project bounds.

    Reprojects polygons to EPSG:3857, buffers 500m, clips global LULC
    raster to buffered bounds, returns path to clipped raster.
    """
    import rasterio
    from rasterio.windows import from_bounds

    gdf_proj = polygons_gdf.to_crs("EPSG:3857")
    buffered = gdf_proj.buffer(500, cap_style=3).to_crs("EPSG:4326")
    xmin, ymin, xmax, ymax = buffered.total_bounds

    output = os.path.join(temp_dir, "lulc_clipped.tif")
    with _rasterio_env(lulc_raster_path):
        with rasterio.open(lulc_raster_path) as src:
            window = from_bounds(xmin, ymin, xmax, ymax, src.transform)
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            profile = src.profile.copy()
            profile.update(
                width=data.shape[1],
                height=data.shape[0],
                transform=transform,
                compress="lzw",
            )
            with rasterio.open(output, "w", **profile) as dst:
                dst.write(data, 1)

    logger.debug(f"Clipped LULC raster to {output}")
    return output


def get_lulc_category(geometry, clipped_lulc_path: str) -> str:
    """Get majority LULC category for a polygon via zonal stats.

    Returns lowercase category name (e.g. 'forest', 'cropland').
    """
    from rasterstats import zonal_stats

    result = zonal_stats(
        geometry,
        clipped_lulc_path,
        all_touched=True,
        stats="count majority",
        nodata=255,
    )
    if result and result[0]["count"] and result[0]["count"] > 0 and result[0]["majority"] is not None:
        majority_int = int(result[0]["majority"])
        for label, values in ESA_LULC_CONVERSIONS.items():
            if majority_int in values:
                return label
    return "forest"


def _clamp_inf(val: float) -> float:
    """Replace infinity with 0 (matching reference behavior)."""
    if math.isinf(val):
        return 0.0
    return val


# -----------------------------------------------------------------------
# Main integration point
# -----------------------------------------------------------------------

def compute_errors(
    results_gdf,
    cfg: PipelineConfig,
    mosaic_path: Optional[str] = None,
    lulc_raster_path: Optional[str] = None,
    polygons_gdf=None,
) -> pd.DataFrame:
    """Add error columns to a TTC results GeoDataFrame.

    Args:
        results_gdf: GeoDataFrame with ``poly_uuid``, ``TTC``, ``area_HA``,
            and ``geometry``.
        cfg: Pipeline config for threshold values.
        mosaic_path: Path to mosaic (needed for shift error). If None,
            shift error is set to 0.
        lulc_raster_path: Path to ESA WorldCover GeoTIFF. If None,
            defaults to 'forest' for all polygons.
        polygons_gdf: Original polygon GeoDataFrame for LULC prep.
            Falls back to results_gdf if not provided.

    Returns:
        DataFrame with added error columns.
    """
    threshold = cfg.zonal.small_sites_area_thresh

    # Prep LULC raster if available (support both local paths and S3 URIs)
    clipped_lulc = None
    if lulc_raster_path and (
        _is_remote_path(lulc_raster_path) or os.path.exists(lulc_raster_path)
    ):
        source_gdf = polygons_gdf if polygons_gdf is not None else results_gdf
        tmp_dir = tempfile.mkdtemp(prefix="ttc_lulc_")
        try:
            clipped_lulc = prep_lulc_data(source_gdf, lulc_raster_path, tmp_dir)
        except Exception as e:
            logger.warning(f"Failed to prep LULC data: {e}. Defaulting to 'forest'.")
    elif lulc_raster_path:
        logger.warning(f"LULC raster not found: {lulc_raster_path}. Defaulting to 'forest'.")

    # Load subregions once
    subregions_gdf = _load_subregions()

    rows = []
    for _, row in results_gdf.iterrows():
        ttc = row["TTC"]
        area = row["area_HA"]
        geom = row.get("geometry", None)

        # Skip flagged/invalid TTC values
        if ttc is None or (isinstance(ttc, float) and np.isnan(ttc)) or ttc == 200.0:
            result_row = row.to_dict()
            result_row.update({
                "shift_error": None,
                "small_site_error": None,
                "lulc_lower_error": None,
                "lulc_upper_error": None,
                "subregion_lower_error": None,
                "subregion_upper_error": None,
                "error_plus": None,
                "error_minus": None,
                "plus_minus_average": None,
            })
            rows.append(result_row)
            continue

        # Per-polygon LULC category
        if clipped_lulc and geom is not None:
            lulc_cat = get_lulc_category(geom, clipped_lulc)
        else:
            lulc_cat = "forest"

        lulc_lo, lulc_hi = lulc_error(ttc, lulc_cat)
        lulc_lo = _clamp_inf(lulc_lo)
        lulc_hi = _clamp_inf(lulc_hi)

        # Per-polygon subregion via spatial join
        if subregions_gdf is not None and geom is not None:
            _, sub_lo, sub_hi = subregion_error(geom, ttc, subregions_gdf)
        else:
            sub_lo, sub_hi = 0.0, 0.0
        sub_lo = _clamp_inf(sub_lo)
        sub_hi = _clamp_inf(sub_hi)

        # Small site error
        ss_err = small_site_error(area, ttc, threshold)

        # Shift error (opt-in, expensive)
        shift_err = 0.0
        if cfg.zonal.shift_error_enabled and mosaic_path and geom is not None:
            try:
                shift_err = shift_error(mosaic_path, geom, ttc)
            except Exception as e:
                logger.warning(f"Shift error failed for polygon: {e}")

        errs = combine_errors(ttc, shift_err, ss_err, lulc_lo, lulc_hi, sub_lo, sub_hi)

        result_row = row.to_dict()
        result_row.update({
            "shift_error": shift_err,
            "small_site_error": ss_err,
            "lulc_lower_error": lulc_lo,
            "lulc_upper_error": lulc_hi,
            "subregion_lower_error": sub_lo,
            "subregion_upper_error": sub_hi,
            **errs,
        })
        rows.append(result_row)

    return pd.DataFrame(rows)
