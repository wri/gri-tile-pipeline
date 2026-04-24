"""Unit tests for per-source tile availability."""

import pytest

from gri_tile_pipeline.storage.tile_paths import (
    ARD_SOURCES,
    raw_ard_keys,
    raw_ard_keys_by_source,
)
from gri_tile_pipeline.tiles.availability import AVAILABLE_SOURCES


def test_raw_ard_keys_by_source_returns_all_sources():
    keys = raw_ard_keys_by_source(2023, 1000, 871)
    assert set(keys.keys()) == set(ARD_SOURCES)
    assert all(isinstance(v, str) and v for v in keys.values())


def test_raw_ard_keys_flat_matches_map_values():
    flat = raw_ard_keys(2023, 1000, 871)
    mapped = raw_ard_keys_by_source(2023, 1000, 871)
    assert set(flat) == set(mapped.values())
    assert len(flat) == len(mapped)


def test_raw_ard_keys_by_source_tile_tag_in_path():
    keys = raw_ard_keys_by_source(2024, 42, 7)
    assert all("42X7Y" in k for k in keys.values())
    assert all(k.startswith("2024/raw/42/7/raw/") for k in keys.values())


def test_available_sources_superset_of_ard():
    assert set(ARD_SOURCES).issubset(AVAILABLE_SOURCES)
    assert "prediction" in AVAILABLE_SOURCES


def test_source_specific_paths_unique():
    keys = raw_ard_keys_by_source(2023, 100, 200)
    # No two sources should resolve to the same key.
    assert len(set(keys.values())) == len(keys)


def test_check_availability_by_source_validates_sources():
    from gri_tile_pipeline.tiles.availability import check_availability_by_source

    with pytest.raises(ValueError, match="Unknown source"):
        check_availability_by_source(
            tiles=[{"year": 2023, "X_tile": 1, "Y_tile": 1, "lon": 0, "lat": 0}],
            dest="s3://nope",
            sources=("bogus",),
        )


def test_check_availability_by_source_empty_tiles():
    from gri_tile_pipeline.tiles.availability import check_availability_by_source

    result = check_availability_by_source(tiles=[], dest="s3://nope")
    assert result == {}
