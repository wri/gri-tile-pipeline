"""Unit tests for tiles/tile_lookup.py."""

from gri_tile_pipeline.tiles.tile_lookup import decode_tile


def test_decode_tile_valid():
    assert decode_tile("1035X727Y") == (1035, 727)


def test_decode_tile_with_spaces():
    assert decode_tile("  1035X727Y  ") == (1035, 727)


def test_decode_tile_negative_coords():
    assert decode_tile("-10X-20Y") == (-10, -20)


def test_decode_tile_invalid_returns_none():
    assert decode_tile("invalid") is None
    assert decode_tile("1035727") is None
    assert decode_tile("") is None
    assert decode_tile("X727Y") is None


def test_decode_tile_zero():
    assert decode_tile("0X0Y") == (0, 0)
