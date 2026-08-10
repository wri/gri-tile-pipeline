"""Tests for streaming mosaic building (gdalbuildvrt/gdal_translate path)."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gri_tile_pipeline.zonal.mosaic import (
    _filter_noise,
    _prefilter_tile,
    build_mosaic_vrt,
)


def _write_tile(path, value, origin_x, origin_y, size=10, pixel_size=0.001):
    """Write a small single-band uint8 GeoTIFF filled with *value*."""
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)
    arr = np.full((size, size), value, dtype=np.uint8)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": size,
        "height": size,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": 255,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return str(path)


class TestFilterNoise:
    def test_weak_isolated_pixel_removed(self):
        # maximum_filter(band, 3) is centered on each pixel (includes
        # itself), so a value's own local max is at least its own value —
        # a genuinely isolated pixel only gets zeroed if it's *itself*
        # below the 30-threshold.
        band = np.zeros((10, 10), dtype=np.uint8)
        band[1, 1] = 25  # below the local-max threshold of 30
        out = _filter_noise(band)
        assert out[1, 1] == 0

    def test_strong_isolated_spike_survives(self):
        # A single pixel at/above the threshold is its own neighborhood
        # maximum, so it passes through even with no supporting neighbors.
        band = np.zeros((10, 10), dtype=np.uint8)
        band[5, 5] = 80
        out = _filter_noise(band)
        assert out[5, 5] == 80

    def test_dense_cluster_kept(self):
        band = np.zeros((10, 10), dtype=np.uint8)
        band[3:7, 3:7] = 80  # solid block should survive the max filter
        out = _filter_noise(band)
        assert out[5, 5] > 0

    def test_dense_region_value_preserved(self):
        # A solid block well above both the neighborhood (30) and value
        # (20) thresholds passes through unchanged: the `<= 0.97` rescale
        # branch only ever fires for a band value of exactly 0 once cast
        # from uint8, so it's a no-op here.
        band = np.full((10, 10), 50, dtype=np.uint8)
        out = _filter_noise(band)
        assert out[5, 5] == 50


class TestPrefilterTile:
    def test_writes_filtered_copy(self, tmp_path):
        src = _write_tile(tmp_path / "in.tif", 80, origin_x=0.0, origin_y=1.0)
        out_dir = tmp_path / "filtered"
        out_dir.mkdir()
        out_path = _prefilter_tile(src, str(out_dir))

        assert out_path != src
        with rasterio.open(out_path) as ds:
            band = ds.read(1)
        # A tile filled solid with 80 is its own dense neighborhood, so it
        # survives the filter unchanged.
        assert band.max() == 80


class TestBuildMosaicVrt:
    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError):
            build_mosaic_vrt([])

    def test_merges_adjacent_tiles(self, tmp_path):
        # Two tiles side by side, each a solid block so noise filtering
        # doesn't zero them out.
        tile_a = _write_tile(tmp_path / "a.tif", 80, origin_x=0.0, origin_y=1.0)
        tile_b = _write_tile(tmp_path / "b.tif", 80, origin_x=0.01, origin_y=1.0)

        output = tmp_path / "mosaic.tif"
        result_path = build_mosaic_vrt([tile_a, tile_b], output_path=str(output))

        assert result_path == str(output)
        with rasterio.open(result_path) as ds:
            assert ds.width >= 20  # combined width of both tiles
            band = ds.read(1)
            assert band.max() > 0

    def test_bounds_clip_output_extent(self, tmp_path):
        tile_a = _write_tile(tmp_path / "a.tif", 80, origin_x=0.0, origin_y=1.0)
        tile_b = _write_tile(tmp_path / "b.tif", 80, origin_x=0.01, origin_y=1.0)

        output = tmp_path / "clipped.tif"
        # Clip to roughly just tile_a's extent.
        bounds = (0.0, 0.99, 0.01, 1.0)
        build_mosaic_vrt([tile_a, tile_b], output_path=str(output), bounds=bounds)

        with rasterio.open(str(output)) as ds:
            b = ds.bounds
            assert b.left == pytest.approx(0.0, abs=1e-6)
            assert b.right == pytest.approx(0.01, abs=1e-6)
