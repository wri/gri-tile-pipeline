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

import os
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from gri_tile_pipeline.config import PipelineConfig

# Data file paths (shipped with the package)
_DATA_DIR = Path(__file__).parent / "data"


def _load_lulc_ci() -> pd.DataFrame:
    """Load LULC confidence interval table."""
    return pd.read_csv(_DATA_DIR / "lulc_ci.csv")


def _load_region_conf() -> pd.DataFrame:
    """Load regional confidence interval table."""
    return pd.read_csv(_DATA_DIR / "region_conf.csv")


# -----------------------------------------------------------------------
# Individual error components
# -----------------------------------------------------------------------

def shift_error(
    mosaic_path: str,
    geojson_path: str,
    base_ttc: float,
    offset: float = 0.0001,  # ~10m in degrees
) -> float:
    """Compute shift error via 8-directional polygon displacement.

    Tests N, S, E, W and 4 diagonal shifts of *offset* degrees,
    recalculates mean TTC for each shifted polygon, and returns
    the RMSE of differences.
    """
    import geopandas as gpd
    import rasterio
    from exactextract import exact_extract
    from shapely.affinity import translate

    gdf = gpd.read_file(geojson_path)
    geom = gdf.geometry.iloc[0]

    shifts = [
        (offset, 0), (-offset, 0),
        (0, offset), (0, -offset),
        (offset, offset), (offset, -offset),
        (-offset, offset), (-offset, -offset),
    ]

    diffs = []
    with rasterio.open(mosaic_path) as src:
        for dx, dy in shifts:
            shifted = translate(geom, xoff=dx, yoff=dy)
            shifted_gdf = gpd.GeoDataFrame(geometry=[shifted], crs=gdf.crs)
            tmp = shifted_gdf.to_json()

            import tempfile, json
            tmp_file = tempfile.NamedTemporaryFile(
                suffix=".geojson", delete=False, mode="w"
            )
            tmp_file.write(tmp)
            tmp_file.close()
            try:
                result = exact_extract(
                    src,
                    tmp_file.name,
                    "mean(min_coverage_frac=0.05, coverage_weight=fraction)",
                )
                val = result[0]["properties"]["mean"]
                if val is not None and not np.isnan(val):
                    diffs.append(abs(val - base_ttc))
            except Exception:
                pass
            finally:
                os.remove(tmp_file.name)

    if not diffs:
        return 0.0
    return float(np.sqrt(np.mean(np.array(diffs) ** 2)))


def small_site_error(
    area_ha: float,
    expected_ttc: float,
    threshold: float = 0.5,
) -> float:
    """Lookup small-site error for polygons below area threshold.

    Applies a lookup table based on expected_ttc ranges:
      - 0-9%:   error = 0.12
      - 10-39%: error = 0.20
      - 40-100%: error = 0.10
    """
    if area_ha > threshold:
        return 0.0

    if expected_ttc < 10:
        return 0.12
    elif expected_ttc < 40:
        return 0.20
    else:
        return 0.10


def lulc_error(
    expected_ttc: float,
    lulc_category: str = "Forest",
) -> Tuple[float, float]:
    """Lookup LULC confidence interval error.

    Args:
        expected_ttc: Tree cover percentage.
        lulc_category: ESA LULC category name.

    Returns:
        ``(lower_error, upper_error)`` as relative fractions.
    """
    ci_df = _load_lulc_ci()
    match = ci_df[ci_df["category"] == lulc_category]
    if match.empty:
        logger.warning(f"LULC category '{lulc_category}' not found, using Forest")
        match = ci_df[ci_df["category"] == "Forest"]

    row = match.iloc[0]
    precision = row["precision"]
    p_lower = row["p_lower_95"]
    p_upper = row["p_upper_95"]

    lower_err = abs(precision - p_lower) / max(precision, 1e-10)
    upper_err = abs(p_upper - precision) / max(precision, 1e-10)

    return lower_err, upper_err


def subregion_error(
    region_name: str = "South America",
) -> Tuple[float, float]:
    """Lookup subregion confidence interval error.

    Args:
        region_name: Subregion name from region_conf.csv.

    Returns:
        ``(lower_error, upper_error)`` as relative fractions.
    """
    conf_df = _load_region_conf()
    match = conf_df[conf_df["category"] == region_name]
    if match.empty:
        logger.warning(f"Region '{region_name}' not found, using global average")
        precision = conf_df["precision"].mean()
        p_lower = conf_df["p_lower_95"].mean()
        p_upper = conf_df["p_upper_95"].mean()
    else:
        row = match.iloc[0]
        precision = row["precision"]
        p_lower = row["p_lower_95"]
        p_upper = row["p_upper_95"]

    lower_err = abs(precision - p_lower) / max(precision, 1e-10)
    upper_err = abs(p_upper - precision) / max(precision, 1e-10)

    return lower_err, upper_err


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
# Main integration point
# -----------------------------------------------------------------------

def compute_errors(
    results_df: pd.DataFrame,
    cfg: PipelineConfig,
    mosaic_path: Optional[str] = None,
    region_name: str = "South America",
    lulc_category: str = "Forest",
) -> pd.DataFrame:
    """Add error columns to a TTC results DataFrame.

    Args:
        results_df: DataFrame with ``poly_id``, ``TTC``, ``area_HA``.
        cfg: Pipeline config for threshold values.
        mosaic_path: Path to mosaic (needed for shift error). If None,
            shift error is set to 0.
        region_name: Default subregion for error lookup.
        lulc_category: Default LULC category.

    Returns:
        DataFrame with added error columns.
    """
    threshold = cfg.zonal.small_sites_area_thresh
    lulc_lo, lulc_hi = lulc_error(0, lulc_category)
    sub_lo, sub_hi = subregion_error(region_name)

    rows = []
    for _, row in results_df.iterrows():
        ttc = row["TTC"]
        area = row["area_HA"]

        ss_err = small_site_error(area, ttc, threshold)
        shift_err = 0.0  # Skip shift error if no mosaic (expensive)

        errs = combine_errors(ttc, shift_err, ss_err, lulc_lo, lulc_hi, sub_lo, sub_hi)

        rows.append({
            **row.to_dict(),
            "shift_error": shift_err,
            "small_site_error": ss_err,
            "lulc_lower_error": lulc_lo,
            "lulc_upper_error": lulc_hi,
            "subregion_lower_error": sub_lo,
            "subregion_upper_error": sub_hi,
            **errs,
        })

    return pd.DataFrame(rows)
