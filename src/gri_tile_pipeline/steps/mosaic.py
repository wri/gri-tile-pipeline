"""Country-scale (or any arbitrary-extent) mosaic step.

Unlike ``zonal_stats.run_zonal_stats`` — which clusters polygons into many
small per-project mosaics and discards each one after extraction — this
step downloads prediction tiles for an already-resolved tile set and merges
them into a single persisted GeoTIFF, streaming the merge via
``build_mosaic_vrt`` so memory stays flat regardless of tile count.
"""

from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger

from gri_tile_pipeline.config import PipelineConfig

TILE_BUFFER_DEG = 1 / 18  # one tile width, matches zonal_stats.py


def run_mosaic(
    tiles: List[Dict[str, Any]],
    dest: str,
    year: int,
    output_path: str,
    cfg: PipelineConfig,
    *,
    polygons_path: str | None = None,
) -> str:
    """Download prediction tiles for *tiles* and merge them into one mosaic.

    Args:
        tiles: Tile dicts (as returned by ``read_tiles_csv``) — the same
            set already used for the download/predict steps.
        dest: Prediction tile root (``s3://bucket/prefix`` or local path).
        year: Prediction year to fetch.
        output_path: Where to write the merged GeoTIFF.
        cfg: PipelineConfig instance.
        polygons_path: Optional polygon file to clip the mosaic's bounds
            to. If omitted, the mosaic covers the full extent of *tiles*.

    Returns:
        Path to the mosaic GeoTIFF.
    """
    import os
    import shutil

    import pandas as pd

    from gri_tile_pipeline.zonal.tile_download import download_prediction_tiles
    from gri_tile_pipeline.zonal.mosaic import build_mosaic_vrt

    if not tiles:
        raise ValueError("No tiles to mosaic")

    tiles_df = pd.DataFrame(tiles)
    tile_bucket = dest.replace("s3://", "").split("/")[0] if dest.startswith("s3://") else dest

    tile_paths = download_prediction_tiles(
        None, tile_bucket, year,
        tiles_df=tiles_df,
        region=cfg.zonal.tile_region,
    )
    if not tile_paths:
        raise RuntimeError(
            f"No prediction tiles available at {dest} for year={year} — "
            "run the download/predict steps first."
        )

    bounds = None
    if polygons_path is not None:
        import geopandas as gpd

        polygons_gdf = gpd.read_file(polygons_path)
        xmin, ymin, xmax, ymax = polygons_gdf.total_bounds
        bounds = (
            xmin - TILE_BUFFER_DEG, ymin - TILE_BUFFER_DEG,
            xmax + TILE_BUFFER_DEG, ymax + TILE_BUFFER_DEG,
        )

    try:
        logger.info(f"Mosaic: merging {len(tile_paths)}/{len(tiles_df)} tiles -> {output_path}")
        mosaic_path = build_mosaic_vrt(tile_paths, output_path=output_path, bounds=bounds)
        logger.info(f"Mosaic written: {mosaic_path}")
        return mosaic_path
    finally:
        tile_dir = os.path.dirname(tile_paths[0])
        if os.path.basename(tile_dir).startswith("ttc_tiles_"):
            shutil.rmtree(tile_dir, ignore_errors=True)
