"""VRT + mosaic building from prediction tiles.

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
) -> str:
    """Build a mosaic GeoTIFF from individual prediction tiles.

    1. Create a GDAL VRT from *tile_paths*
    2. Translate VRT to a single compressed GeoTIFF
    3. Apply noise filtering (morphological max + threshold)

    Args:
        tile_paths: Local paths to prediction GeoTIFF tiles.
        output_path: Desired output path. If None, uses a temp file.

    Returns:
        Path to the mosaic GeoTIFF.
    """
    from osgeo import gdal
    import rasterio
    import scipy.ndimage

    if not tile_paths:
        raise ValueError("No tile paths provided")

    if output_path is None:
        output_path = os.path.join(
            tempfile.mkdtemp(prefix="ttc_mosaic_"), "mosaic.tif"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Build VRT
    vrt_path = output_path.replace(".tif", ".vrt")
    gdal.BuildVRT(
        vrt_path,
        tile_paths,
        options=gdal.BuildVRTOptions(srcNodata=255, VRTNodata=255),
    )

    # Step 2: Translate to GeoTIFF
    raw_path = output_path.replace(".tif", "_raw.tif")
    ds = gdal.Open(vrt_path)
    translate_opts = gdal.TranslateOptions(
        gdal.ParseCommandLine(
            "-ot Byte -co COMPRESS=LZW -a_nodata 255 -co BIGTIFF=YES"
        )
    )
    gdal.Translate(raw_path, ds, options=translate_opts)
    ds = None

    # Clean up VRT
    try:
        os.remove(vrt_path)
    except OSError:
        pass

    # Step 3: Noise filtering
    with rasterio.open(raw_path) as src:
        meta = src.meta.copy()
        arr = src.read(1).copy()

    # Morphological max filter + threshold
    arr_filtered = scipy.ndimage.maximum_filter(arr, 3)
    arr_float = arr.astype(np.float32)
    arr_float[arr_float <= 0.97] = arr_float[arr_float <= 0.97] / 0.97
    arr_float[arr_filtered < 30] = 0.0
    arr_float[arr_float < 20] = 0.0
    arr_out = arr_float.astype(np.uint8)

    meta.update({"nodata": 255})
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(arr_out, 1)

    # Clean up raw
    try:
        os.remove(raw_path)
    except OSError:
        pass

    logger.info(f"Mosaic built: {output_path} ({arr_out.shape})")
    return output_path
