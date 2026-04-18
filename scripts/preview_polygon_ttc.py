#!/usr/bin/env python
"""Preview a polygon overlaid on its TTC prediction tiles.

Usage:
    uv run python scripts/preview_polygon_ttc.py <poly_uuid>
    uv run python scripts/preview_polygon_ttc.py <poly_uuid> --year 2021
    uv run python scripts/preview_polygon_ttc.py <poly_uuid> --show-shifts
    uv run python scripts/preview_polygon_ttc.py <poly_uuid> -o output.png
"""
from __future__ import annotations

import argparse
import sys
import tempfile

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from loguru import logger
from matplotlib.colors import Normalize
from rasterio.merge import merge
from shapely import from_wkb

from shapely.affinity import translate

from gri_tile_pipeline.config import load_config
from gri_tile_pipeline.zonal.tile_download import (
    download_prediction_tiles,
    load_tile_lookup,
    pre_filter_tiles,
)

# ~10m in degrees, matching error_propagation.py
SHIFT_OFFSET = 0.0001081081
SHIFT_DIRECTIONS = {
    "N":  (0, SHIFT_OFFSET),
    "S":  (0, -SHIFT_OFFSET),
    "E":  (SHIFT_OFFSET, 0),
    "W":  (-SHIFT_OFFSET, 0),
    "NE": (SHIFT_OFFSET, SHIFT_OFFSET),
    "NW": (-SHIFT_OFFSET, SHIFT_OFFSET),
    "SE": (SHIFT_OFFSET, -SHIFT_OFFSET),
    "SW": (-SHIFT_OFFSET, -SHIFT_OFFSET),
}

GEOPARQUET = "temp/tm.geoparquet"


def load_polygon(poly_uuid: str, geoparquet: str = GEOPARQUET) -> tuple[gpd.GeoDataFrame, int]:
    """Load a single polygon by poly_uuid, return (GeoDataFrame, default_year)."""
    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")

    df = con.sql(f"""
        SELECT *, ST_AsWKB(geom) AS __wkb
        FROM read_parquet('{geoparquet}')
        WHERE poly_uuid = '{poly_uuid}'
    """).df()

    if df.empty:
        logger.error(f"No polygon found for poly_uuid='{poly_uuid}'")
        sys.exit(1)

    df["geometry"] = df["__wkb"].apply(lambda b: from_wkb(bytes(b)))
    gdf = gpd.GeoDataFrame(df.drop(columns=["geom", "__wkb"]), geometry="geometry", crs="EPSG:4326")

    plantstart = gdf["plantstart"].iloc[0]
    year = plantstart.year - 1 if plantstart else 2020
    return gdf, year


def render_preview(
    gdf: gpd.GeoDataFrame,
    tile_paths: list[str],
    year: int,
    poly_uuid: str,
    output: str,
    show_shifts: bool = False,
) -> None:
    """Render the TTC tiles cropped to the polygon extent and overlay the boundary."""
    # Compute view bounds: polygon bbox + 10% buffer
    minx, miny, maxx, maxy = gdf.total_bounds
    dx = (maxx - minx) or 0.001
    dy = (maxy - miny) or 0.001
    pad = max(dx, dy) * 0.10
    view_bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    # Merge tiles clipped to view bounds
    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        arr, transform = merge(datasets, bounds=view_bounds, nodata=255)
    finally:
        for ds in datasets:
            ds.close()

    band = arr[0].astype(np.float32)
    band[band == 255] = np.nan  # nodata → transparent

    # Compute image extent for imshow
    h, w = band.shape
    extent = [
        transform.c,                    # left
        transform.c + transform.a * w,  # right
        transform.f + transform.e * h,  # bottom
        transform.f,                    # top
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(
        band,
        extent=extent,
        cmap="viridis",
        norm=Normalize(vmin=0, vmax=100),
        interpolation="nearest",
    )
    gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=2, label="Original")

    if show_shifts:
        for label, (dx, dy) in SHIFT_DIRECTIONS.items():
            shifted_geoms = gdf.geometry.apply(lambda g: translate(g, xoff=dx, yoff=dy))
            shifted_gdf = gpd.GeoDataFrame(geometry=shifted_geoms, crs=gdf.crs)
            shifted_gdf.boundary.plot(
                ax=ax, edgecolor="orange", linewidth=0.5, alpha=0.3,
            )
        # Single legend entry for all shifts
        from matplotlib.lines import Line2D
        ax.legend(
            handles=[
                Line2D([], [], color="red", linewidth=2, label="Original"),
                Line2D([], [], color="orange", linewidth=0.5, alpha=0.3, label="Shift envelope (~10 m)"),
            ],
            loc="upper right",
        )

    ax.set_xlim(view_bounds[0], view_bounds[2])
    ax.set_ylim(view_bounds[1], view_bounds[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"TTC Prediction — {poly_uuid} ({year})")
    fig.colorbar(im, ax=ax, label="Tree Extent (%)", shrink=0.7)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Preview saved → {output}")


def main():
    parser = argparse.ArgumentParser(description="Preview polygon with TTC prediction tiles")
    parser.add_argument("poly_uuid", help="poly_uuid from tm.geoparquet")
    parser.add_argument("--year", type=int, default=None, help="Prediction year (default: plantstart - 1)")
    parser.add_argument("--geoparquet", default=GEOPARQUET, help="Path to polygon geoparquet")
    parser.add_argument("--show-shifts", action="store_true",
                        help="Overlay 8-directional ~10m shift boundaries (shift error visualization)")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    gdf, derived_year = load_polygon(args.poly_uuid, args.geoparquet)
    year = args.year or derived_year
    output = args.output or f"temp/{args.poly_uuid}_ttc_preview.png"

    logger.info(f"Polygon: {args.poly_uuid}, year={year}")

    cfg = load_config()
    lookup = load_tile_lookup(parquet_path=cfg.zonal.lookup_parquet, lookup_csv=cfg.zonal.lookup_csv)

    # Find intersecting tiles
    combined = gdf.geometry.union_all()
    tiles_df = pre_filter_tiles(combined, lookup)
    if tiles_df.empty:
        logger.error("No tiles intersect this polygon")
        sys.exit(1)
    logger.info(f"Found {len(tiles_df)} intersecting tile(s)")

    # Download prediction tiles
    tile_paths = download_prediction_tiles(
        polygons_path=None,
        tile_bucket=cfg.zonal.tile_bucket,
        year=year,
        tiles_df=tiles_df,
        region=cfg.zonal.tile_region,
    )
    if not tile_paths:
        logger.error("No prediction tiles available for this polygon/year")
        sys.exit(1)

    render_preview(gdf, tile_paths, year, args.poly_uuid, output, show_shifts=args.show_shifts)


if __name__ == "__main__":
    main()
