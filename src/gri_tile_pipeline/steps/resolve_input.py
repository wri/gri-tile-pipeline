"""Resolve various input formats to a list of tiles for pipeline processing.

Supported input types:
  - **tiles CSV**: standard format with Year, X, Y, X_tile, Y_tile columns
  - **request CSV**: project_id, plantstart_year columns -> query geoparquet -> tiles
  - **polygon file**: GeoJSON, GeoPackage, etc. -> spatial join against tile grid
  - **JSON request**: legacy inbound JSON format -> DuckDB tile resolution
"""

from __future__ import annotations

import csv as csv_mod
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class ResolvedInput:
    """Result of resolving an input file to tiles."""

    tiles: List[Dict[str, Any]]
    input_type: str  # tiles_csv, request_csv, polygon_file, json_request
    polygons_gdf: Any = None  # Optional GeoDataFrame
    polygons_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


TILES_CSV_COLUMNS = {"Year", "X", "Y", "X_tile", "Y_tile"}
REQUEST_CSV_COLUMNS = {"project_id", "plantstart_year"}


def detect_input_type(path: str) -> str:
    """Detect the input file type from extension and column headers."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        return "json_request"

    if ext in (".geojson", ".gpkg", ".shp"):
        return "polygon_file"

    if ext == ".csv":
        with open(path, newline="") as f:
            reader = csv_mod.DictReader(f)
            cols = set(reader.fieldnames or [])

        if TILES_CSV_COLUMNS.issubset(cols):
            return "tiles_csv"
        if REQUEST_CSV_COLUMNS.issubset(cols):
            return "request_csv"

        raise ValueError(
            f"CSV '{path}' doesn't match tiles format {sorted(TILES_CSV_COLUMNS)} "
            f"or request format {sorted(REQUEST_CSV_COLUMNS)}. "
            f"Found columns: {sorted(cols)}"
        )

    if ext in (".parquet", ".geoparquet"):
        return "polygon_file"

    # Default: try as polygon file
    return "polygon_file"


def resolve_to_tiles(
    input_path: str,
    cfg,
    *,
    year: int | None = None,
    year_from_plantstart: bool = False,
    geoparquet: str | None = None,
) -> ResolvedInput:
    """Resolve any supported input format to a list of tiles.

    Args:
        input_path: Path to the input file.
        cfg: PipelineConfig instance.
        year: Explicit prediction year (required for polygon file input).
        year_from_plantstart: Derive year as plantstart - 1 from polygon data.
        geoparquet: Path to TerraMatch geoparquet (for request CSV input).

    Returns:
        ResolvedInput with tiles list and optional polygon data.
    """
    input_type = detect_input_type(input_path)
    logger.info(f"Detected input type: {input_type} for {input_path}")

    if input_type == "tiles_csv":
        return _resolve_tiles_csv(input_path)

    if input_type == "json_request":
        return _resolve_json_request(input_path, cfg)

    if input_type == "request_csv":
        geoparquet = geoparquet or "temp/tm.geoparquet"
        return _resolve_request_csv(input_path, geoparquet, cfg)

    if input_type == "polygon_file":
        return _resolve_polygon_file(
            input_path, cfg, year=year, year_from_plantstart=year_from_plantstart,
        )

    raise ValueError(f"Unsupported input type: {input_type}")


# ---------------------------------------------------------------------------
# Per-type resolvers
# ---------------------------------------------------------------------------


def _resolve_tiles_csv(path: str) -> ResolvedInput:
    from gri_tile_pipeline.tiles.csv_io import read_tiles_csv

    tiles = read_tiles_csv(path)
    logger.info(f"Read {len(tiles)} tiles from {path}")
    return ResolvedInput(tiles=tiles, input_type="tiles_csv")


def _resolve_json_request(path: str, cfg) -> ResolvedInput:
    import tempfile

    from gri_tile_pipeline.steps.ingest import run_ingest
    from gri_tile_pipeline.tiles.csv_io import read_tiles_csv

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", prefix="gri_ingest_", delete=False)
    tmp.close()
    try:
        n = run_ingest(path, cfg.parquet_path, tmp.name)
        if n == 0:
            return ResolvedInput(tiles=[], input_type="json_request")
        tiles = read_tiles_csv(tmp.name)
        return ResolvedInput(tiles=tiles, input_type="json_request")
    finally:
        os.unlink(tmp.name)


def _resolve_request_csv(path: str, geoparquet: str, cfg) -> ResolvedInput:
    from gri_tile_pipeline.steps.project_e2e import _extract_from_request_csv
    from gri_tile_pipeline.tiles.tile_lookup import identify_tiles_for_polygons

    gdf, geojson_path, meta = _extract_from_request_csv(path, geoparquet)
    tiles = identify_tiles_for_polygons(
        gdf,
        lookup_parquet=cfg.zonal.lookup_parquet or cfg.parquet_path,
        lookup_csv=cfg.zonal.lookup_csv,
    )
    logger.info(f"Resolved {len(tiles)} tiles from {meta['n_polygons']} polygons")
    return ResolvedInput(
        tiles=tiles,
        input_type="request_csv",
        polygons_gdf=gdf,
        polygons_path=str(geojson_path),
        metadata=meta,
    )


def _resolve_polygon_file(
    path: str, cfg, *, year: int | None, year_from_plantstart: bool
) -> ResolvedInput:
    import geopandas as gpd
    import pandas as pd

    from gri_tile_pipeline.tiles.tile_lookup import identify_tiles_for_polygons

    gdf = gpd.read_file(path)

    if year_from_plantstart:
        if "plantstart" not in gdf.columns:
            raise ValueError(
                "--year-from-plantstart requires a 'plantstart' column in the polygon file"
            )
        gdf["pred_year"] = gdf["plantstart"].apply(
            lambda ps: ps.year - 1 if pd.notna(ps) and hasattr(ps, "year") else None
        )
        invalid = gdf["pred_year"].isna()
        if invalid.all():
            raise ValueError("No valid plantstart dates found in polygon file")
        if invalid.any():
            logger.warning(f"Dropping {invalid.sum()} polygon(s) with no valid plantstart")
            gdf = gdf[~invalid].copy()
        gdf["pred_year"] = gdf["pred_year"].astype(int)
    elif year is not None:
        gdf["pred_year"] = year
    else:
        raise ValueError(
            "Polygon file input requires --year or --year-from-plantstart"
        )

    tiles = identify_tiles_for_polygons(
        gdf,
        lookup_parquet=cfg.zonal.lookup_parquet or cfg.parquet_path,
        lookup_csv=cfg.zonal.lookup_csv,
    )
    logger.info(f"Resolved {len(tiles)} tiles from {len(gdf)} polygons")
    return ResolvedInput(
        tiles=tiles,
        input_type="polygon_file",
        polygons_gdf=gdf,
        polygons_path=path,
        metadata={"n_polygons": len(gdf)},
    )
