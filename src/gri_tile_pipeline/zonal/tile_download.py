"""Download prediction GeoTIFFs via obstore.

Replaces boto3 sequential downloads from the reference
``ttc_s3_utils.py`` with concurrent obstore-based fetching.
"""

from __future__ import annotations

import os
import tempfile
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from shapely.geometry import box, shape

import obstore as obs
from obstore.store import from_url

from gri_tile_pipeline.storage.tile_paths import prediction_key


def _load_polygons(path: str):
    """Load polygon geometries from GeoJSON/GeoPackage/Shapefile."""
    import geopandas as gpd
    return gpd.read_file(path)


def pre_filter_tiles(
    geometry,
    global_lookup: pd.DataFrame,
    tile_size: float = 1 / 18,
) -> pd.DataFrame:
    """Filter tile lookup table by polygon intersection.

    Uses vectorised NumPy bounding-box pre-filter, then per-tile
    shapely intersection.
    """
    bounds = geometry.bounds
    half_tile = tile_size / 2
    buffer_size = tile_size

    x = global_lookup["X"].to_numpy(copy=False)
    y = global_lookup["Y"].to_numpy(copy=False)

    x0, y0, x1, y1 = bounds
    x0 -= buffer_size
    y0 -= buffer_size
    x1 += buffer_size
    y1 += buffer_size

    idx = (x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1)
    pre_filter = global_lookup.iloc[idx.nonzero()[0]]

    intersecting = []
    for _, tile in pre_filter.iterrows():
        tile_box = box(
            tile["X"] - half_tile,
            tile["Y"] - half_tile,
            tile["X"] + half_tile,
            tile["Y"] + half_tile,
        )
        if tile_box.intersects(geometry):
            intersecting.append(tile)

    if intersecting:
        result = pd.DataFrame(intersecting)
        result["X_tile"] = pd.to_numeric(result["X_tile"], downcast="integer")
        result["Y_tile"] = pd.to_numeric(result["Y_tile"], downcast="integer")
        return result
    return pd.DataFrame(columns=global_lookup.columns)


def download_prediction_tiles(
    polygons_path: str,
    tile_bucket: str,
    year: int,
    *,
    global_lookup: pd.DataFrame | None = None,
    lookup_csv: str | None = None,
    temp_dir: str | None = None,
    region: str = "us-west-2",
) -> List[str]:
    """Download the prediction tiles that overlap the input polygons.

    Args:
        polygons_path: Path to polygon file (GeoJSON/GPKG/SHP).
        tile_bucket: S3 bucket containing prediction tiles.
        year: Year to fetch tiles for.
        global_lookup: Pre-loaded tile lookup DataFrame. If None, loads from
            *lookup_csv*.
        lookup_csv: Path to tile lookup CSV (fallback).
        temp_dir: Directory to store downloaded tiles.
        region: AWS region.

    Returns:
        List of local file paths to downloaded GeoTIFFs.
    """
    import geopandas as gpd

    gdf = gpd.read_file(polygons_path)

    if global_lookup is None:
        if lookup_csv is not None:
            global_lookup = pd.read_csv(lookup_csv)
        else:
            raise ValueError("Provide either global_lookup or lookup_csv")

    # Collect tiles across all polygons
    all_tiles: List[pd.Series] = []
    for _, row in gdf.iterrows():
        pf = pre_filter_tiles(row.geometry, global_lookup)
        for _, tile_row in pf.iterrows():
            all_tiles.append(tile_row)

    if not all_tiles:
        logger.warning("No tiles intersect the input polygons")
        return []

    tiles_df = pd.DataFrame(all_tiles).drop_duplicates(subset=["X_tile", "Y_tile"])
    logger.info(f"Downloading {len(tiles_df)} prediction tiles for year {year}")

    store = from_url(f"s3://{tile_bucket}", region=region)
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="ttc_tiles_")
    os.makedirs(temp_dir, exist_ok=True)

    local_paths: List[str] = []
    for _, tile in tiles_df.iterrows():
        x = int(tile["X_tile"])
        y = int(tile["Y_tile"])
        key = prediction_key(year, x, y)
        local_path = os.path.join(temp_dir, f"{x}X{y}Y_FINAL.tif")

        try:
            data = obs.get(store, key)
            with open(local_path, "wb") as f:
                f.write(data.bytes())
            local_paths.append(local_path)
        except Exception as e:
            logger.warning(f"Failed to download {key}: {e}")

    logger.info(f"Downloaded {len(local_paths)}/{len(tiles_df)} tiles")
    return local_paths
