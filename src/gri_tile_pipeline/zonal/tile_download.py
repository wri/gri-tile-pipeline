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

from gri_tile_pipeline.storage.tile_paths import prediction_key


def _load_polygons(path: str):
    """Load polygon geometries from GeoJSON/GeoPackage/Shapefile."""
    import geopandas as gpd
    return gpd.read_file(path)


def load_tile_lookup(
    parquet_path: str | None = None,
    lookup_csv: str | None = None,
) -> pd.DataFrame:
    """Load tile lookup table from parquet or CSV.

    The lookup table must have columns: X, Y, X_tile, Y_tile.

    Args:
        parquet_path: Path to tiledb.parquet (preferred).
        lookup_csv: Path to CSV fallback.

    Returns:
        DataFrame with tile coordinates.
    """
    if parquet_path and os.path.exists(parquet_path):
        import duckdb
        df = duckdb.sql(f"SELECT X, Y, X_tile, Y_tile FROM '{parquet_path}'").df()
        logger.info(f"Loaded tile lookup from {parquet_path}: {len(df)} tiles")
        return df
    elif lookup_csv and os.path.exists(lookup_csv):
        df = pd.read_csv(lookup_csv)
        logger.info(f"Loaded tile lookup from {lookup_csv}: {len(df)} tiles")
        return df
    else:
        raise FileNotFoundError(
            f"Tile lookup not found. Tried parquet={parquet_path}, csv={lookup_csv}"
        )


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
    polygons_path: str | None,
    tile_bucket: str,
    year: int,
    *,
    tiles_df: pd.DataFrame | None = None,
    global_lookup: pd.DataFrame | None = None,
    lookup_parquet: str | None = None,
    lookup_csv: str | None = None,
    temp_dir: str | None = None,
    region: str = "us-east-1",
) -> List[str]:
    """Download the prediction tiles that overlap the input polygons.

    Args:
        polygons_path: Path to polygon file (GeoJSON/GPKG/SHP).  Can be
            ``None`` when *tiles_df* is provided.
        tile_bucket: S3 bucket containing prediction tiles.
        year: Year to fetch tiles for.
        tiles_df: Pre-computed tile DataFrame with columns
            ``X_tile, Y_tile, X, Y``.  When provided, the polygon
            intersection step is skipped entirely.
        global_lookup: Pre-loaded tile lookup DataFrame. If None, loads from
            *lookup_parquet* or *lookup_csv*.
        lookup_parquet: Path to tiledb.parquet (preferred over CSV).
        lookup_csv: Path to tile lookup CSV (fallback).
        temp_dir: Directory to store downloaded tiles.
        region: AWS region.

    Returns:
        List of local file paths to downloaded GeoTIFFs.
    """
    if tiles_df is not None:
        # Tiles already identified (e.g. by spatial clustering step)
        tiles_dedup = tiles_df.drop_duplicates(subset=["X_tile", "Y_tile"])
    else:
        import geopandas as gpd

        gdf = gpd.read_file(polygons_path)

        if global_lookup is None:
            global_lookup = load_tile_lookup(
                parquet_path=lookup_parquet, lookup_csv=lookup_csv
            )

        # Collect tiles across all polygons
        all_tiles: List[pd.Series] = []
        for _, row in gdf.iterrows():
            pf = pre_filter_tiles(row.geometry, global_lookup)
            for _, tile_row in pf.iterrows():
                all_tiles.append(tile_row)

        if not all_tiles:
            logger.warning("No tiles intersect the input polygons")
            return []

        tiles_dedup = pd.DataFrame(all_tiles).drop_duplicates(subset=["X_tile", "Y_tile"])
    logger.info(f"Downloading {len(tiles_dedup)} prediction tiles for year {year}")

    from gri_tile_pipeline.storage.obstore_utils import from_dest
    if tile_bucket.startswith(("s3://", "/", ".")):
        store = from_dest(tile_bucket, region=region)
    else:
        store = from_dest(f"s3://{tile_bucket}", region=region)
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="ttc_tiles_")
    os.makedirs(temp_dir, exist_ok=True)

    local_paths: List[str] = []
    for _, tile in tiles_dedup.iterrows():
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

    logger.info(f"Downloaded {len(local_paths)}/{len(tiles_dedup)} tiles")
    return local_paths
