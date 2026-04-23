"""Operational tests: tile availability checking against local directories."""

import os

import pytest

from gri_tile_pipeline.storage.tile_paths import raw_ard_keys


def test_ard_files_match_expected_keys(ard_dir):
    """Verify that example ARD directory has all expected files."""
    keys = raw_ard_keys(2024, 1000, 871)
    expected_basenames = set()
    for k in keys:
        parts = k.split("/")
        # e.g. "2024/raw/1000/871/raw/misc/dem_1000X871Y.hkl"
        subdir = parts[-2]  # misc, s1, s2_10, s2_20, clouds
        basename = parts[-1]
        expected_basenames.add((subdir, basename))

    for subdir, basename in expected_basenames:
        path = os.path.join(str(ard_dir), subdir, basename)
        assert os.path.isfile(path), f"Missing ARD file: {path}"
