#!/usr/bin/env python
"""Extract a single polygon from the TerraMatch geoparquet and run zonal stats.

Usage:
    uv run python scripts/run_single_polygon_stats.py GHA_22_INEC
    uv run python scripts/run_single_polygon_stats.py GHA_22_INEC --year 2021
    uv run python scripts/run_single_polygon_stats.py GHA_22_INEC --lulc-raster reference/LCCS-Map-300m-2020-v2.1.1.tif

Without --year, it defaults to plantstart.year - 1 (matching the reference logic).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import duckdb
import geopandas as gpd
from loguru import logger

from gri_tile_pipeline.config import load_config
from gri_tile_pipeline.steps.zonal_stats import run_zonal_stats

GEOPARQUET = "temp/tm.geoparquet"


def extract_polygon(short_name: str, geoparquet: str = GEOPARQUET) -> tuple[Path, int]:
    """Extract polygons for *short_name* to a temp GeoJSON, return (path, year)."""
    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")

    df = con.sql(f"""
        SELECT *, ST_AsWKB(geom) AS __wkb
        FROM read_parquet('{geoparquet}')
        WHERE short_name = '{short_name}'
    """).df()

    if df.empty:
        logger.error(f"No polygons found for short_name='{short_name}'")
        sys.exit(1)

    from shapely import from_wkb

    df["geometry"] = df["__wkb"].apply(lambda b: from_wkb(bytes(b)))
    gdf = gpd.GeoDataFrame(df.drop(columns=["geom", "__wkb"]), geometry="geometry", crs="EPSG:4326")

    # Derive year from plantstart (year - 1), matching reference logic
    plantstart = gdf["plantstart"].iloc[0]
    year = plantstart.year - 1 if plantstart else 2020

    out = Path(tempfile.mkdtemp(prefix="ttc_single_")) / f"{short_name}.geojson"
    gdf.to_file(out, driver="GeoJSON")

    logger.info(f"Extracted {len(gdf)} polygon(s) for {short_name} → {out}")
    logger.info(f"  country={gdf['country'].iloc[0]}, plantstart={plantstart}, year={year}")
    logger.info(f"  area={gdf['calc_area'].iloc[0]:.2f} ha")
    return out, year


def main():
    parser = argparse.ArgumentParser(description="Run zonal stats for a single project from tm.geoparquet")
    parser.add_argument("short_name", help="Project short_name (e.g. GHA_22_INEC)")
    parser.add_argument("--year", type=int, default=None, help="Override prediction year")
    parser.add_argument("--lulc-raster", default="reference/LCCS-Map-300m-2020-v2.1.1.tif",
                        help="Path or S3 URI to LULC raster")
    parser.add_argument("--geoparquet", default=GEOPARQUET, help="Path to polygon geoparquet")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    geojson_path, derived_year = extract_polygon(args.short_name, args.geoparquet)
    year = args.year or derived_year
    output = args.output or f"temp/{args.short_name}_stats.csv"

    cfg = load_config()
    if args.lulc_raster:
        cfg.zonal.lulc_raster_path = args.lulc_raster

    logger.info(f"Running zonal stats: year={year}, lulc={cfg.zonal.lulc_raster_path}")
    run_zonal_stats(
        str(geojson_path),
        cfg.zonal.tile_bucket,
        year,
        output,
        cfg,
    )
    logger.info(f"Done → {output}")


if __name__ == "__main__":
    main()
