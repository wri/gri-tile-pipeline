"""Unit tests for preprocessing/cloud_removal.py."""

import numpy as np

from gri_tile_pipeline.preprocessing.cloud_removal import (
    align_interp_array_lr,
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


class TestAlignInterpArrayLrEarlyExits:
    """Guards ported from reference sentinel-tree-cover 0654ad6: return the raw
    mosaic instead of failing (empty pool) or fitting on too few samples."""

    T, H, W, DATE = 3, 20, 20, 1

    def _inputs(self, interp, water_mask):
        rng = np.random.default_rng(0)
        array = rng.uniform(0.05, 0.5, (self.T, self.H, self.W, 10)).astype(np.float32)
        mosaic = rng.uniform(0.05, 0.5, (self.H, self.W, 10)).astype(np.float32)
        interp_array = mosaic[np.newaxis].copy()
        return interp_array, array, interp, mosaic, water_mask

    def test_no_fully_clear_pixels_in_window_returns_raw_mosaic(self):
        # Every pixel is water==1: it passes the "<= 1" land check but fails
        # the "< 1" training requirement, so the training pool is empty.
        # Previously np.percentile([]) raised IndexError.
        interp = np.full((self.T, self.H, self.W), 0.5, np.float32)
        interp[self.DATE, :10] = 0.0
        water = np.ones((self.H, self.W), np.float32)
        ia, arr, it, mos, wm = self._inputs(interp, water)
        out, to_remove = align_interp_array_lr(ia, arr, self.DATE, it, mos, wm, seed=42)
        np.testing.assert_array_equal(out, mos)
        assert to_remove == []

    def test_fewer_than_100_samples_returns_raw_mosaic(self):
        interp = np.ones((self.T, self.H, self.W), np.float32)
        interp[self.DATE] = 0.5
        interp[self.DATE].flat[:50] = 0.0  # only 50 fully-clear training pixels
        water = np.zeros((self.H, self.W), np.float32)
        ia, arr, it, mos, wm = self._inputs(interp, water)
        out, to_remove = align_interp_array_lr(ia, arr, self.DATE, it, mos, wm, seed=42)
        np.testing.assert_array_equal(out, mos)
        assert to_remove == []

    def test_enough_samples_still_fits_regression(self):
        interp = np.ones((self.T, self.H, self.W), np.float32)
        interp[self.DATE] = 0.0
        interp[self.DATE, :5] = 0.5  # 100 cloudy pixels to fill, 300 clear to train on
        water = np.zeros((self.H, self.W), np.float32)
        ia, arr, it, mos, wm = self._inputs(interp, water)
        out, _ = align_interp_array_lr(ia, arr, self.DATE, it, mos, wm, seed=42)
        assert out.shape == mos.shape
        assert not np.array_equal(out[:5], mos[:5])  # cloudy rows were re-fitted
        np.testing.assert_array_equal(out[5:], mos[5:])  # clear rows untouched
