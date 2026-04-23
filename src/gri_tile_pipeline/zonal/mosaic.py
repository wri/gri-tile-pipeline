"""Mosaic building from prediction tiles using rasterio.

Ported from reference ``ttc_s3_utils.py`` ``build_vrt`` / ``make_mosaic``.
"""

from __future__ import annotations

import os
import tempfile
from typing import List

import numpy as np
from loguru import logger


def build_mosaic(
    tile_paths: List[str],
    output_path: str | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> str:
    """Build a mosaic GeoTIFF from individual prediction tiles.

    1. Merge tiles via rasterio.merge (optionally clipped to *bounds*)
    2. Apply noise filtering (morphological max + threshold)

    Args:
        tile_paths: Local paths to prediction GeoTIFF tiles.
        output_path: Desired output path. If None, uses a temp file.
        bounds: Optional (xmin, ymin, xmax, ymax) to restrict the merge
            to only the needed region. Significantly reduces memory and
            processing time when polygons cover a small fraction of the
            tile extent.

    Returns:
        Path to the mosaic GeoTIFF.
    """
    import rasterio
    from rasterio.merge import merge
    import scipy.ndimage

    if not tile_paths:
        raise ValueError("No tile paths provided")

    if output_path is None:
        output_path = os.path.join(
            tempfile.mkdtemp(prefix="ttc_mosaic_"), "mosaic.tif"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Merge tiles (clipped to bounds if provided)
    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        merge_kwargs = {"nodata": 255}
        if bounds is not None:
            merge_kwargs["bounds"] = bounds
        arr, transform = merge(datasets, **merge_kwargs)
    finally:
        for ds in datasets:
            ds.close()

    # arr shape is (bands, height, width) — we only need band 1
    band = arr[0].copy()

    # Step 2: Noise filtering (matching reference make_mosaic)
    arr_filtered = scipy.ndimage.maximum_filter(band, 3)
    arr_float = band.astype(np.float32)
    arr_float[arr_float <= 0.97] = arr_float[arr_float <= 0.97] / 0.97
    arr_float[arr_filtered < 30] = 0.0
    arr_float[arr_float < 20] = 0.0
    arr_out = arr_float.astype(np.uint8)

    # Write output
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": arr_out.shape[1],
        "height": arr_out.shape[0],
        "count": 1,
        "crs": datasets[0].crs if datasets else "EPSG:4326",
        "transform": transform,
        "nodata": 255,
        "compress": "lzw",
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(arr_out, 1)

    logger.info(f"Mosaic built: {output_path} ({arr_out.shape})")
    return output_path
