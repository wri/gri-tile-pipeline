"""Tree cover mean per polygon via exactextract.

Ported from reference ``tree_cover_indicator.py``.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger


def compute_ttc(
    polygons_path: str,
    mosaic_path: str,
) -> pd.DataFrame:
    """Calculate mean tree cover for each polygon.

    Uses ``exactextract`` with ``mean(min_coverage_frac=0.05,
    coverage_weight=fraction)`` to compute area-weighted tree cover
    percentage per polygon.

    Args:
        polygons_path: Path to polygon file (GeoJSON/GPKG/SHP).
        mosaic_path: Path to mosaic GeoTIFF from :func:`build_mosaic`.

    Returns:
        DataFrame with columns: ``poly_id``, ``TTC``, ``area_HA``,
        ``geometry`` (if available).
    """
    import geopandas as gpd
    import rasterio
    from exactextract import exact_extract

    gdf = gpd.read_file(polygons_path)

    # Explode MultiPolygon geometries into individual Polygon rows
    if gdf.geometry.geom_type.isin(["MultiPolygon"]).any():
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        logger.info(f"Exploded MultiPolygons: {len(gdf)} features")

    # Ensure an ID column exists
    if "poly_id" not in gdf.columns:
        gdf["poly_id"] = range(len(gdf))

    # Calculate area in hectares (project to Mollweide)
    gdf_proj = gdf.to_crs("ESRI:54009")
    gdf["area_HA"] = gdf_proj.geometry.area / 10_000

    results: List[Dict[str, Any]] = []

    with rasterio.open(mosaic_path) as src:
        for idx, row in gdf.iterrows():
            geom = row.geometry
            poly_id = row["poly_id"]
            area_ha = row["area_HA"]

            # Write single feature to temp GeoJSON for exactextract
            tmp_geojson = tempfile.NamedTemporaryFile(
                suffix=".geojson", delete=False, mode="w"
            )
            feature = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"],
                        "properties": {"id": str(poly_id)},
                    }
                ],
            }
            json.dump(feature, tmp_geojson)
            tmp_geojson.close()

            try:
                result = exact_extract(
                    src,
                    tmp_geojson.name,
                    "mean(min_coverage_frac=0.05, coverage_weight=fraction)",
                )
                ttc = result[0]["properties"]["mean"]
                if ttc is None or np.isnan(ttc):
                    ttc = 0.0
            except Exception as e:
                logger.warning(f"exactextract failed for polygon {poly_id}: {e}")
                ttc = 0.0
            finally:
                os.remove(tmp_geojson.name)

            results.append({
                "poly_id": poly_id,
                "TTC": ttc,
                "area_HA": area_ha,
            })

    logger.info(f"Computed TTC for {len(results)} polygons")
    return pd.DataFrame(results)
