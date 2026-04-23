"""Render a preview image of a polygon overlaid on its TTC prediction tiles.

Matplotlib is imported lazily so `preview_polygon` can be imported in
environments without a plotting backend (the import only fails when
`render_preview` is actually called).
"""

from __future__ import annotations

import sys

from gri_tile_pipeline.duckdb_utils import connect_with_spatial


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


def load_polygon(poly_uuid: str, geoparquet: str):
    """Load a single polygon by poly_uuid, return (GeoDataFrame, default_year).

    default_year is plantstart - 1 if available, else 2020.
    """
    import geopandas as gpd
    from shapely import from_wkb

    con = connect_with_spatial()
    try:
        df = con.sql(f"""
            SELECT *, ST_AsWKB(geom) AS __wkb
            FROM read_parquet('{geoparquet}')
            WHERE poly_uuid = '{poly_uuid}'
        """).df()
    finally:
        con.close()

    if df.empty:
        raise ValueError(f"No polygon found for poly_uuid='{poly_uuid}'")

    df["geometry"] = df["__wkb"].apply(lambda b: from_wkb(bytes(b)))
    gdf = gpd.GeoDataFrame(
        df.drop(columns=["geom", "__wkb"]), geometry="geometry", crs="EPSG:4326",
    )

    plantstart = gdf["plantstart"].iloc[0]
    year = plantstart.year - 1 if plantstart else 2020
    return gdf, year


def render_preview(
    gdf,
    tile_paths: list[str],
    year: int,
    poly_uuid: str,
    output: str,
    show_shifts: bool = False,
) -> None:
    """Render the TTC tiles cropped to the polygon extent and overlay the boundary."""
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    from matplotlib.colors import Normalize
    from rasterio.merge import merge
    from shapely.affinity import translate
    import geopandas as gpd

    minx, miny, maxx, maxy = gdf.total_bounds
    dx = (maxx - minx) or 0.001
    dy = (maxy - miny) or 0.001
    pad = max(dx, dy) * 0.10
    view_bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        arr, transform = merge(datasets, bounds=view_bounds, nodata=255)
    finally:
        for ds in datasets:
            ds.close()

    band = arr[0].astype(np.float32)
    band[band == 255] = np.nan

    h, w = band.shape
    extent = [
        transform.c,
        transform.c + transform.a * w,
        transform.f + transform.e * h,
        transform.f,
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(
        band, extent=extent, cmap="viridis",
        norm=Normalize(vmin=0, vmax=100), interpolation="nearest",
    )
    gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=2, label="Original")

    if show_shifts:
        for _label, (dx, dy) in SHIFT_DIRECTIONS.items():
            shifted_geoms = gdf.geometry.apply(lambda g: translate(g, xoff=dx, yoff=dy))
            shifted_gdf = gpd.GeoDataFrame(geometry=shifted_geoms, crs=gdf.crs)
            shifted_gdf.boundary.plot(
                ax=ax, edgecolor="orange", linewidth=0.5, alpha=0.3,
            )
        from matplotlib.lines import Line2D
        ax.legend(
            handles=[
                Line2D([], [], color="red", linewidth=2, label="Original"),
                Line2D([], [], color="orange", linewidth=0.5, alpha=0.3,
                       label="Shift envelope (~10 m)"),
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


def preview_polygon(
    poly_uuid: str,
    cfg,
    *,
    year: int | None = None,
    geoparquet: str = "temp/tm.geoparquet",
    output: str | None = None,
    show_shifts: bool = False,
) -> str:
    """End-to-end: load polygon, fetch intersecting tiles, render preview.

    Returns the output path written.
    """
    from gri_tile_pipeline.zonal.tile_download import (
        download_prediction_tiles,
        load_tile_lookup,
        pre_filter_tiles,
    )

    gdf, derived_year = load_polygon(poly_uuid, geoparquet)
    year = year or derived_year
    output = output or f"temp/{poly_uuid}_ttc_preview.png"

    lookup = load_tile_lookup(
        parquet_path=cfg.zonal.lookup_parquet, lookup_csv=cfg.zonal.lookup_csv,
    )
    combined = gdf.geometry.union_all()
    tiles_df = pre_filter_tiles(combined, lookup)
    if tiles_df.empty:
        raise ValueError("No tiles intersect this polygon")

    tile_paths = download_prediction_tiles(
        polygons_path=None,
        tile_bucket=cfg.zonal.tile_bucket,
        year=year,
        tiles_df=tiles_df,
        region=cfg.zonal.tile_region,
    )
    if not tile_paths:
        raise ValueError("No prediction tiles available for this polygon/year")

    render_preview(gdf, tile_paths, year, poly_uuid, output, show_shifts=show_shifts)
    return output
