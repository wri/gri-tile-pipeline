"""Mosaic building from prediction tiles using rasterio.

Ported from reference ``ttc_s3_utils.py`` ``build_vrt`` / ``make_mosaic``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import List

import numpy as np
from loguru import logger

from gri_tile_pipeline.zonal.ttc_colormap import build_ttc_colormap


def _filter_noise(band: np.ndarray) -> np.ndarray:
    """Apply morphological-max + threshold noise filtering to a TTC band.

    Matches the reference ``make_mosaic`` cleanup: drop pixels with no
    high-confidence neighborhood, and rescale everything below the
    saturation threshold.
    """
    import scipy.ndimage

    arr_filtered = scipy.ndimage.maximum_filter(band, 3)
    arr_float = band.astype(np.float32)
    arr_float[arr_float <= 0.97] = arr_float[arr_float <= 0.97] / 0.97
    arr_float[arr_filtered < 30] = 0.0
    arr_float[arr_float < 20] = 0.0
    return arr_float.astype(np.uint8)


def build_mosaic(
    tile_paths: List[str],
    output_path: str | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> str:
    """Build a mosaic GeoTIFF from individual prediction tiles.

    1. Merge tiles via rasterio.merge (optionally clipped to *bounds*)
    2. Apply noise filtering (morphological max + threshold)

    Loads the full merged array into memory — fine for the handful of
    tiles in a single polygon cluster, but not for country-scale areas.
    See :func:`build_mosaic_vrt` for a streaming alternative.

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
    arr_out = _filter_noise(arr[0])

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


def _prefilter_tile(tile_path: str, out_dir: str) -> str:
    """Apply :func:`_filter_noise` to a single tile, writing a filtered copy.

    Filtering per-tile (rather than on a merged country-sized array) keeps
    peak memory at O(1 tile) regardless of how many tiles are being
    mosaicked, and avoids seam artifacts that windowed post-merge filtering
    would introduce at window boundaries.
    """
    import rasterio

    out_path = os.path.join(out_dir, os.path.basename(tile_path))
    with rasterio.open(tile_path) as src:
        band = src.read(1)
        profile = src.profile

    filtered = _filter_noise(band)
    profile.update(dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(filtered, 1)

    return out_path


def _apply_ttc_colormap(path: str) -> None:
    """Attach the legacy TTC green-ramp color table to band 1 of *path*."""
    import rasterio

    with rasterio.open(path, "r+") as dst:
        dst.write_colormap(1, build_ttc_colormap())


def build_mosaic_vrt(
    tile_paths: List[str],
    output_path: str | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> str:
    """Build a mosaic GeoTIFF from prediction tiles without materializing
    the full merged array in memory.

    Unlike :func:`build_mosaic` (which uses ``rasterio.merge`` and loads
    everything at once), this filters tiles individually before merging so
    peak memory stays at O(1 tile) regardless of how many tiles are being
    combined. The output has the legacy TTC green-ramp color table attached
    (see :mod:`gri_tile_pipeline.zonal.ttc_colormap`), matching the visual
    style of the reference country mosaics.

    Args:
        tile_paths: Local paths to prediction GeoTIFF tiles.
        output_path: Desired output path. If None, uses a temp file.
        bounds: Optional (xmin, ymin, xmax, ymax) to clip the mosaic to.

    Returns:
        Path to the mosaic GeoTIFF.
    """
    import rasterio
    from rasterio.merge import merge

    if not tile_paths:
        raise ValueError("No tile paths provided")

    if output_path is None:
        output_path = os.path.join(
            tempfile.mkdtemp(prefix="ttc_mosaic_"), "mosaic.tif"
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    work_dir = tempfile.mkdtemp(prefix="ttc_mosaic_vrt_")
    try:
        filtered_dir = os.path.join(work_dir, "filtered")
        os.makedirs(filtered_dir, exist_ok=True)
        filtered_paths = [_prefilter_tile(p, filtered_dir) for p in tile_paths]

        datasets = [rasterio.open(p) for p in filtered_paths]
        try:
            merge_kwargs: dict = {"nodata": 255}
            if bounds is not None:
                merge_kwargs["bounds"] = bounds
            arr, transform = merge(datasets, **merge_kwargs)
            profile = datasets[0].profile
        finally:
            for ds in datasets:
                ds.close()

        profile.update(
            dtype="uint8",
            width=arr.shape[2],
            height=arr.shape[1],
            transform=transform,
            nodata=255,
            compress="lzw",
            tiled=True,
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(arr)
        _apply_ttc_colormap(output_path)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    logger.info(f"Mosaic built: {output_path} ({len(tile_paths)} tiles)")
    return output_path
