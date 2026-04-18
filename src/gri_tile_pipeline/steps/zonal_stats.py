"""Zonal statistics step: calculate tree cover per polygon.

Runs locally (not on Lambda).  Polygons are spatially clustered by
shared prediction tiles so that each cluster gets its own compact
mosaic, avoiding enormous rasters for geographically scattered projects.
"""

from __future__ import annotations

import os
import shutil
from typing import List

import pandas as pd
from loguru import logger

from gri_tile_pipeline.config import PipelineConfig


def _cleanup_batch(tile_paths: List[str], mosaic_path: str) -> None:
    """Remove temp mosaic and tile files from a completed batch."""
    # Remove mosaic temp dir
    mosaic_dir = os.path.dirname(mosaic_path)
    if mosaic_dir and os.path.basename(mosaic_dir).startswith("ttc_mosaic_"):
        shutil.rmtree(mosaic_dir, ignore_errors=True)

    # Remove tile download temp dir
    if tile_paths:
        tile_dir = os.path.dirname(tile_paths[0])
        if tile_dir and os.path.basename(tile_dir).startswith("ttc_tiles_"):
            shutil.rmtree(tile_dir, ignore_errors=True)


def run_zonal_stats(
    polygons: str,
    tiles_bucket: str,
    year: int,
    output: str,
    cfg: PipelineConfig,
    *,
    lookup_parquet: str | None = None,
    lookup_csv: str | None = None,
    include_cols: list[str] | None = None,
) -> None:
    """Calculate zonal tree cover statistics for *polygons*.

    1. Load tile lookup (parquet or CSV)
    2. Cluster polygons by shared tiles (DuckDB spatial join)
    3. Per cluster: download tiles → build mosaic → exactextract → errors
    4. Concatenate and write results CSV
    """
    import geopandas as gpd

    from gri_tile_pipeline.zonal.tile_download import (
        download_prediction_tiles,
        load_tile_lookup,
    )
    from gri_tile_pipeline.zonal.mosaic import build_mosaic
    from gri_tile_pipeline.zonal.exactextract_ttc import compute_ttc
    from gri_tile_pipeline.zonal.error_propagation import compute_errors
    from gri_tile_pipeline.zonal.spatial_clustering import cluster_polygons_by_tiles

    logger.info(f"Running zonal stats for {polygons}, year={year}")

    # Resolve tile lookup source
    parquet = lookup_parquet or cfg.zonal.lookup_parquet
    csv = lookup_csv or cfg.zonal.lookup_csv
    lookup = load_tile_lookup(parquet_path=parquet, lookup_csv=csv)

    polygons_gdf = gpd.read_file(polygons)

    # Ensure globally unique poly_uuid before clustering
    if "poly_uuid" not in polygons_gdf.columns:
        polygons_gdf["poly_uuid"] = range(len(polygons_gdf))

    # Cluster polygons by shared tiles
    clusters = cluster_polygons_by_tiles(polygons_gdf, lookup)
    n_clusters = len(clusters)

    if n_clusters == 0:
        logger.error("No prediction tiles intersect the polygons — cannot compute stats")
        return

    logger.info(
        f"Spatial clustering: {n_clusters} cluster(s) from "
        f"{len(polygons_gdf)} polygons"
    )

    lulc_path = cfg.zonal.lulc_raster_path or None
    tile_buf = 1 / 18  # one tile width buffer for noise filter

    all_results: list[pd.DataFrame] = []

    for i, (cluster_gdf, cluster_tiles_df) in enumerate(clusters, 1):
        logger.info(
            f"Batch {i}/{n_clusters}: "
            f"{len(cluster_gdf)} polygons, {len(cluster_tiles_df)} tiles"
        )

        # Download tiles for this cluster
        tile_paths = download_prediction_tiles(
            None, tiles_bucket, year,
            tiles_df=cluster_tiles_df,
            region=cfg.zonal.tile_region,
        )

        if not tile_paths:
            logger.warning(f"Batch {i}: no tiles downloaded, skipping")
            continue

        # Compute cluster-local mosaic bounds
        xmin, ymin, xmax, ymax = cluster_gdf.total_bounds
        mosaic_bounds = (
            xmin - tile_buf, ymin - tile_buf,
            xmax + tile_buf, ymax + tile_buf,
        )

        mosaic_path = build_mosaic(tile_paths, bounds=mosaic_bounds)

        try:
            # Compute TTC
            batch_results = compute_ttc(
                None, mosaic_path,
                include_cols=include_cols,
                polygons_gdf=cluster_gdf,
            )

            # Error propagation
            batch_results = compute_errors(
                batch_results, cfg,
                mosaic_path=mosaic_path,
                lulc_raster_path=lulc_path,
            )

            all_results.append(batch_results)
            logger.info(f"Batch {i}/{n_clusters}: {len(batch_results)} results")
        finally:
            _cleanup_batch(tile_paths, mosaic_path)

    if not all_results:
        logger.error("No batches produced results")
        return

    results = pd.concat(all_results, ignore_index=True)

    # Write output (drop geometry column for CSV)
    output_cols = [c for c in results.columns if c != "geometry"]
    results[output_cols].to_csv(output, index=False)
    logger.info(f"Results written to {output} ({len(results)} polygons)")
