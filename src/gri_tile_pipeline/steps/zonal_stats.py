"""Zonal statistics step: calculate tree cover per polygon.

Runs locally (not on Lambda).
"""

from __future__ import annotations

from loguru import logger

from gri_tile_pipeline.config import PipelineConfig


def run_zonal_stats(
    polygons: str,
    tiles_bucket: str,
    year: int,
    output: str,
    cfg: PipelineConfig,
) -> None:
    """Calculate zonal tree cover statistics for *polygons*.

    1. Load polygon geometries
    2. Filter prediction tiles by intersection
    3. Download prediction GeoTIFFs via obstore
    4. Build VRT mosaic
    5. Run exactextract for each polygon
    6. Calculate error propagation
    7. Write results CSV
    """
    from gri_tile_pipeline.zonal.tile_download import download_prediction_tiles
    from gri_tile_pipeline.zonal.mosaic import build_mosaic
    from gri_tile_pipeline.zonal.exactextract_ttc import compute_ttc
    from gri_tile_pipeline.zonal.error_propagation import compute_errors

    logger.info(f"Running zonal stats for {polygons}, year={year}")

    # Step 1-3: Download tiles
    tile_paths = download_prediction_tiles(polygons, tiles_bucket, year)

    # Step 4: Build mosaic
    mosaic_path = build_mosaic(tile_paths)

    # Step 5-6: Compute TTC and errors
    results = compute_ttc(polygons, mosaic_path)
    results = compute_errors(results, cfg)

    # Step 7: Write output
    results.to_csv(output, index=False)
    logger.info(f"Results written to {output} ({len(results)} polygons)")
