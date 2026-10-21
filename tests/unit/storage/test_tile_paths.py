"""Unit tests for storage/tile_paths.py."""

from gri_tile_pipeline.storage.tile_paths import prediction_key, raw_ard_keys


def test_raw_ard_keys_count():
    keys = raw_ard_keys(2024, 1000, 871)
    assert len(keys) == 6


def test_raw_ard_keys_names():
    keys = raw_ard_keys(2024, 1000, 871)
    basenames = [k.split("/")[-1] for k in keys]
    assert "dem_1000X871Y.hkl" in basenames
    assert "1000X871Y.hkl" in basenames  # s1 or s2
    assert "s2_dates_1000X871Y.hkl" in basenames
    assert "clouds_1000X871Y.hkl" in basenames


def test_raw_ard_keys_prefix():
    keys = raw_ard_keys(2024, 1000, 871)
    for k in keys:
        assert k.startswith("2024/raw/1000/871/raw/")


def test_raw_ard_keys_subdirs():
    keys = raw_ard_keys(2024, 1000, 871)
    subdirs = {k.split("/")[-2] for k in keys}
    assert subdirs == {"misc", "s1", "s2_10", "s2_20", "clouds"}


def test_prediction_key_format():
    key = prediction_key(2024, 1000, 871)
    assert key == "2024/tiles/1000/871/1000X871Y_FINAL.tif"


def test_prediction_key_different_tile():
    key = prediction_key(2025, 500, 300)
    assert key == "2025/tiles/500/300/500X300Y_FINAL.tif"
