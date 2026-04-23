"""Unit tests for storage/stac.py and the COG writer profile."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from gri_tile_pipeline.storage.stac import (
    COG_MEDIA_TYPE,
    STAC_VERSION,
    bbox_to_geometry,
    build_predict_stac_item,
    tile_bbox,
)
from gri_tile_pipeline.storage.tile_paths import (
    prediction_key,
    prediction_stac_key,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def test_prediction_stac_key_is_sibling_of_prediction_key():
    tif = prediction_key(2024, 1000, 871)
    jsn = prediction_stac_key(2024, 1000, 871)
    assert tif.rsplit(".", 1)[0] == jsn.rsplit(".", 1)[0]
    assert jsn.endswith(".json")
    assert tif.endswith(".tif")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def test_tile_bbox_at_origin():
    west, south, east, north = tile_bbox(0.0, 0.0)
    assert west == pytest.approx(-1.0 / 36.0)
    assert east == pytest.approx(1.0 / 36.0)
    assert south == pytest.approx(-1.0 / 36.0)
    assert north == pytest.approx(1.0 / 36.0)


def test_tile_bbox_offset():
    west, south, east, north = tile_bbox(10.0, -5.0)
    span = 1.0 / 18.0
    assert east - west == pytest.approx(span)
    assert north - south == pytest.approx(span)
    assert (west + east) / 2 == pytest.approx(10.0)
    assert (south + north) / 2 == pytest.approx(-5.0)


def test_bbox_to_geometry_closed_ring():
    geom = bbox_to_geometry((0.0, 0.0, 1.0, 1.0))
    assert geom["type"] == "Polygon"
    ring = geom["coordinates"][0]
    assert len(ring) == 5
    assert ring[0] == ring[-1]
    # lon-lat order
    assert ring[0] == [0.0, 0.0]
    assert ring[2] == [1.0, 1.0]


# ---------------------------------------------------------------------------
# STAC item builder
# ---------------------------------------------------------------------------

def _sample_item() -> dict:
    return build_predict_stac_item(
        x_tile=1000,
        y_tile=871,
        year=2024,
        lon=10.0,
        lat=-5.0,
        asset_href="1000X871Y_FINAL.tif",
        model={
            "name": "predict_graph-172",
            "path": "/tmp/models/predict_graph-172.pb",
            "sha256": "abc123",
            "input_size": 172,
            "output_size": 158,
            "length": 4,
        },
        pipeline_version="0.1.0",
        git_sha="deadbeef",
        run_id="a1b2c3d4",
        created="2026-04-21T14:30:00Z",
    )


def test_stac_item_core_fields():
    item = _sample_item()
    assert item["type"] == "Feature"
    assert item["stac_version"] == STAC_VERSION
    assert item["id"] == "1000X871Y_2024"
    assert item["collection"] == "gri-tree-cover-predictions"


def test_stac_item_bbox_matches_geometry():
    item = _sample_item()
    bbox = item["bbox"]
    ring = item["geometry"]["coordinates"][0]
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    assert min(xs) == pytest.approx(bbox[0])
    assert min(ys) == pytest.approx(bbox[1])
    assert max(xs) == pytest.approx(bbox[2])
    assert max(ys) == pytest.approx(bbox[3])


def test_stac_item_temporal_is_year_span():
    item = _sample_item()
    props = item["properties"]
    assert props["datetime"] is None
    assert props["start_datetime"] == "2024-01-01T00:00:00Z"
    assert props["end_datetime"] == "2024-12-31T23:59:59Z"


def test_stac_item_model_properties():
    item = _sample_item()
    props = item["properties"]
    assert props["gri:model_name"] == "predict_graph-172"
    assert props["gri:model_sha256"] == "abc123"
    assert props["gri:model_input_size"] == 172
    assert props["gri:model_output_size"] == 158
    assert props["gri:model_length"] == 4
    assert props["gri:pipeline_version"] == "0.1.0"
    assert props["gri:git_sha"] == "deadbeef"
    assert props["gri:run_id"] == "a1b2c3d4"


def test_stac_item_asset_is_cog():
    item = _sample_item()
    data = item["assets"]["data"]
    assert data["href"] == "1000X871Y_FINAL.tif"
    assert data["type"] == COG_MEDIA_TYPE
    assert "data" in data["roles"]


def test_stac_item_is_json_serializable():
    item = _sample_item()
    dumped = json.dumps(item, sort_keys=True)
    restored = json.loads(dumped)
    assert restored == item


def test_stac_item_allows_nullable_provenance():
    item = build_predict_stac_item(
        x_tile=1, y_tile=2, year=2024, lon=0.0, lat=0.0,
        asset_href="x.tif",
        model={"name": "n", "path": "p", "sha256": None,
               "input_size": 172, "output_size": 158, "length": 4},
        pipeline_version="unknown",
        git_sha=None,
        run_id=None,
        created="2026-04-21T00:00:00Z",
    )
    props = item["properties"]
    assert props["gri:git_sha"] is None
    assert props["gri:run_id"] is None
    assert props["gri:model_sha256"] is None


# ---------------------------------------------------------------------------
# COG profile assertions on the writer
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    pytest.importorskip("rasterio", reason="rasterio required") is None,
    reason="rasterio required",
)
def test_cog_writer_produces_tiled_overviewed_geotiff():
    import rasterio

    # The writer is defined in loaders/predict_tile.py. Import lazily so we
    # don't force TF into the test process.
    from loaders.predict_tile import _cog_write

    h = w = 640  # large enough that blocksize=256 yields >=1 overview level
    arr = np.random.default_rng(0).integers(0, 101, size=(h, w), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cog.tif")
        _cog_write(path, arr, lon=10.0, lat=-5.0)

        with rasterio.open(path) as ds:
            assert ds.driver in ("GTiff", "COG")
            assert ds.count == 1
            assert ds.dtypes[0] == "uint8"
            assert ds.nodata == 255
            assert ds.crs.to_epsg() == 4326
            # COG stores internal tiling; block shape reflects that.
            bh, bw = ds.block_shapes[0]
            assert bh == 256 and bw == 256, f"expected 256x256 blocks, got {bh}x{bw}"
            # Overviews should be present for a 640-px tile at blocksize 256.
            assert len(ds.overviews(1)) >= 1
