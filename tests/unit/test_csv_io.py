"""Unit tests for tiles/csv_io.py."""

import os
import tempfile

import pytest

from gri_tile_pipeline.tiles.csv_io import read_tiles_csv, write_tiles_csv


@pytest.fixture
def sample_tiles():
    return [
        {"year": 2024, "lon": -73.5, "lat": 45.5, "X_tile": 1000, "Y_tile": 871},
        {"year": 2024, "lon": -73.6, "lat": 45.6, "X_tile": 1001, "Y_tile": 872},
    ]


def test_write_then_read_roundtrip(sample_tiles, tmp_path):
    csv_path = str(tmp_path / "tiles.csv")
    write_tiles_csv(csv_path, sample_tiles)
    result = read_tiles_csv(csv_path)

    assert len(result) == 2
    assert result[0]["year"] == 2024
    assert result[0]["lon"] == -73.5
    assert result[0]["lat"] == 45.5
    assert result[0]["X_tile"] == 1000
    assert result[0]["Y_tile"] == 871


def test_missing_columns_raises(tmp_path):
    csv_path = str(tmp_path / "bad.csv")
    with open(csv_path, "w") as f:
        f.write("Year,X,Y\n")
        f.write("2024,-73.5,45.5\n")

    with pytest.raises(ValueError, match="missing columns"):
        read_tiles_csv(csv_path)


def test_empty_csv(tmp_path):
    csv_path = str(tmp_path / "empty.csv")
    with open(csv_path, "w") as f:
        f.write("Year,X,Y,Y_tile,X_tile\n")

    result = read_tiles_csv(csv_path)
    assert result == []


def test_csv_header_format(sample_tiles, tmp_path):
    """CSV headers are Year,X,Y,Y_tile,X_tile (not the dict keys)."""
    csv_path = str(tmp_path / "tiles.csv")
    write_tiles_csv(csv_path, sample_tiles)

    with open(csv_path) as f:
        header = f.readline().strip()
    assert header == "Year,X,Y,Y_tile,X_tile"
