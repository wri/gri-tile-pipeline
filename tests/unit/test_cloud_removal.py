"""Unit tests for preprocessing/cloud_removal.py."""

import numpy as np
import pytest

from gri_tile_pipeline.preprocessing.cloud_removal import (
    id_missing_px,
    interpolate_na_vals,
)


class TestIdMissingPx:
    def test_all_zero_detected(self):
        """Timesteps with all-zero pixels should be flagged."""
        x = np.random.default_rng(42).uniform(0.01, 0.5, (5, 32, 32, 10)).astype(
            np.float32
        )
        x[2] = 0.0  # Entire timestep is zero
        missing = id_missing_px(x)
        assert 2 in missing

    def test_clean_data_no_missing(self):
        x = np.random.default_rng(42).uniform(0.01, 0.5, (5, 32, 32, 10)).astype(
            np.float32
        )
        missing = id_missing_px(x)
        assert len(missing) == 0

    def test_returns_array(self):
        x = np.random.default_rng(42).uniform(0.01, 0.5, (5, 32, 32, 10)).astype(
            np.float32
        )
        missing = id_missing_px(x)
        assert isinstance(missing, np.ndarray)

    def test_saturated_detected(self):
        """Timesteps with many pixels at 1.0 should be flagged."""
        x = np.random.default_rng(42).uniform(0.01, 0.5, (5, 32, 32, 10)).astype(
            np.float32
        )
        x[1] = 1.0  # Entire timestep saturated
        missing = id_missing_px(x)
        assert 1 in missing


class TestInterpolateNaVals:
    def test_nan_replaced(self):
        x = np.random.default_rng(42).uniform(0.1, 0.5, (5, 8, 8, 10)).astype(
            np.float32
        )
        x[2, 3, 3, :] = np.nan
        result = interpolate_na_vals(x)
        assert not np.any(np.isnan(result))

    def test_non_nan_preserved(self):
        x = np.random.default_rng(42).uniform(0.1, 0.5, (5, 8, 8, 10)).astype(
            np.float32
        )
        original = x.copy()
        result = interpolate_na_vals(x)
        np.testing.assert_array_equal(result, original)

    def test_all_nan_timestep(self):
        """If entire timestep is NaN, should still not produce NaN (replaced with 0)."""
        x = np.random.default_rng(42).uniform(0.1, 0.5, (3, 4, 4, 4)).astype(
            np.float32
        )
        x[1] = np.nan
        result = interpolate_na_vals(x)
        assert not np.any(np.isnan(result))
