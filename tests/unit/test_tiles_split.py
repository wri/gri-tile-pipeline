"""Unit tests for tiles/split.py."""

from pathlib import Path

import pytest

from gri_tile_pipeline.tiles.split import (
    DEFAULT_CHUNK_LADDER,
    STEADY_CHUNK_SIZE,
    split_csv,
)


def _write_csv(path: Path, n_rows: int) -> None:
    with path.open("w") as f:
        f.write("Year,X,Y,X_tile,Y_tile\n")
        for i in range(n_rows):
            f.write(f"2023,{i},{i},{i},{i}\n")


def test_split_ladder_default(tmp_path):
    src = tmp_path / "big.csv"
    # 100 + 200 + 400 + 50 = 750 -> four chunks of sizes 100, 200, 400, 50
    _write_csv(src, 750)
    files = split_csv(str(src))
    assert len(files) == 4
    sizes = [sum(1 for _ in open(f)) - 1 for f in files]  # subtract header
    assert sizes == [100, 200, 400, 50]


def test_split_fixed_chunk_size(tmp_path):
    src = tmp_path / "even.csv"
    _write_csv(src, 205)
    files = split_csv(str(src), chunk_size=100)
    assert len(files) == 3
    sizes = [sum(1 for _ in open(f)) - 1 for f in files]
    assert sizes == [100, 100, 5]


def test_split_empty_file(tmp_path):
    src = tmp_path / "empty.csv"
    src.write_text("")
    files = split_csv(str(src))
    assert files == []


def test_split_preserves_header(tmp_path):
    src = tmp_path / "h.csv"
    _write_csv(src, 50)
    files = split_csv(str(src), chunk_size=25)
    for f in files:
        with open(f) as fh:
            assert fh.readline().strip() == "Year,X,Y,X_tile,Y_tile"


def test_split_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        split_csv(str(tmp_path / "nope.csv"))


def test_ladder_constants_unchanged():
    assert DEFAULT_CHUNK_LADDER == [100, 200, 400, 800]
    assert STEADY_CHUNK_SIZE == 1600
