"""Tree cover mean per polygon via exactextract.

Ported from reference ``tree_cover_indicator.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger


def compute_ttc(
    polygons_path: str | None,
    mosaic_path: str,
    include_cols: List[str] | None = None,
    *,
    polygons_gdf: "gpd.GeoDataFrame | None" = None,
) -> "gpd.GeoDataFrame":
    """Calculate mean tree cover for each polygon.

    Uses ``exactextract`` with ``mean(min_coverage_frac=0.05,
    coverage_weight=fraction)`` to compute area-weighted tree cover
    percentage per polygon.

    Args:
        polygons_path: Path to polygon file (GeoJSON/GPKG/SHP).  Can be
            ``None`` when *polygons_gdf* is provided.
        mosaic_path: Path to mosaic GeoTIFF from :func:`build_mosaic`.
        include_cols: Extra columns from the source polygons to carry
            through to the output (e.g. ``["unique_id", "FARM_ID"]``).
            If ``None``, auto-detects common ID columns.
        polygons_gdf: Pre-loaded GeoDataFrame.  When provided,
            *polygons_path* is ignored.

    Returns:
        GeoDataFrame with columns: ``poly_uuid``, ``TTC``, ``area_HA``,
        any *include_cols*, and ``geometry``.
    """
    import geopandas as gpd
    from exactextract import exact_extract

    if polygons_gdf is not None:
        gdf = polygons_gdf.copy()
    else:
        gdf = gpd.read_file(polygons_path)

    # Fix invalid (self-intersecting) geometries, keeping them as single
    # Polygons. buffer(0) resolves self-intersections; if that produces a
    # MultiPolygon from what was originally a Polygon, keep only the largest
    # part (the sliver is an artifact). Legitimate MultiPolygons are exploded.
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        logger.info(f"Repairing {invalid_mask.sum()} invalid geometries")
        was_polygon = gdf.geometry.geom_type == "Polygon"
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(
            lambda g: g.buffer(0)
        )
        # For originally-Polygon rows that became MultiPolygon, keep largest
        became_multi = was_polygon & (gdf.geometry.geom_type == "MultiPolygon")
        if became_multi.any():
            gdf.loc[became_multi, "geometry"] = gdf.loc[became_multi, "geometry"].apply(
                lambda g: max(g.geoms, key=lambda p: p.area)
            )
            logger.info(f"Kept largest part for {became_multi.sum()} repaired polygons")

    # Drop geometries that are empty or collapsed to non-polygon types after repair
    degenerate = gdf.geometry.is_empty | ~gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    if degenerate.any():
        bad_ids = gdf.loc[degenerate, "poly_uuid"].tolist() if "poly_uuid" in gdf.columns else []
        for bid in bad_ids:
            logger.warning(f"Dropping degenerate geometry for poly_uuid={bid}")
        if not bad_ids:
            logger.warning(f"Dropping {degenerate.sum()} degenerate geometries")
        gdf = gdf[~degenerate].reset_index(drop=True)

    # Explode legitimate MultiPolygon geometries into individual Polygon rows
    if gdf.geometry.geom_type.isin(["MultiPolygon"]).any():
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        logger.info(f"Exploded MultiPolygons: {len(gdf)} features")

    # Auto-detect ID columns if none specified
    _COMMON_ID_COLS = ["unique_id", "FARM_ID", "PLOT_ID", "FARMER_ID", "id", "ID", "fid", "FID", "name", "NAME"]
    if include_cols is None:
        include_cols = [c for c in _COMMON_ID_COLS if c in gdf.columns]

    # Ensure an ID column exists
    if "poly_uuid" not in gdf.columns:
        gdf["poly_uuid"] = range(len(gdf))

    # Calculate area in hectares (project to Mollweide)
    gdf_proj = gdf.to_crs("ESRI:54009")
    gdf["area_HA"] = gdf_proj.geometry.area / 10_000

    # Force 2D geometry (exactextract may choke on 3D/Z geometries)
    from shapely import force_2d
    gdf["geometry"] = gdf["geometry"].apply(force_2d)

    # Build a minimal GeoDataFrame with only the columns exactextract needs
    extract_gdf = gpd.GeoDataFrame(
        {"poly_uuid": gdf["poly_uuid"].astype(str)},
        geometry=gdf.geometry,
        crs=gdf.crs,
    )

    # Run exactextract on all polygons at once
    ee_results = exact_extract(
        mosaic_path,
        extract_gdf,
        "mean(min_coverage_frac=0.05, coverage_weight=fraction)",
        include_cols=["poly_uuid"],
        output="pandas",
    )

    # Join back with area and geometry
    results: List[Dict[str, Any]] = []
    for _, ee_row in ee_results.iterrows():
        pid = ee_row["poly_uuid"]
        ttc = ee_row["mean"]

        # Find matching source row
        match = gdf[gdf["poly_uuid"].astype(str) == str(pid)]
        if match.empty:
            continue
        src_row = match.iloc[0]

        if ttc is None or (isinstance(ttc, float) and np.isnan(ttc)):
            logger.warning(f"NaN TTC for polygon {pid}, flagging with 200")
            ttc = 200.0

        row_data = {
            "poly_uuid": pid,
            "TTC": ttc,
            "area_HA": src_row["area_HA"],
        }
        for col in include_cols:
            if col in src_row.index:
                row_data[col] = src_row[col]
        row_data["geometry"] = src_row.geometry
        results.append(row_data)

    if not results:
        logger.warning("No polygons produced TTC results")
        cols = ["poly_uuid", "TTC", "area_HA"] + list(include_cols) + ["geometry"]
        return gpd.GeoDataFrame(
            columns=cols,
            geometry="geometry",
            crs=gdf.crs,
        )

    logger.info(f"Computed TTC for {len(results)} polygons")
    return gpd.GeoDataFrame(results, geometry="geometry", crs=gdf.crs)
