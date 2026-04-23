"""Unit tests for preprocessing/temporal_resampling.py."""

import numpy as np
import pytest

from gri_tile_pipeline.preprocessing.temporal_resampling import resample_to_biweekly


class TestResampleToBiweekly:
    def test_always_24_outputs(self, sample_s2_array):
        """Output always has T=24 regardless of input count."""
        dates = np.linspace(0, 345, sample_s2_array.shape[0])
        result, _ = resample_to_biweekly(sample_s2_array, dates)
        assert result.shape[0] == 24

    def test_constant_input_passthrough(self):
        """Constant input should produce (near-)constant output."""
        x = np.full((10, 8, 8, 4), 0.42, dtype=np.float32)
        dates = np.linspace(10, 340, 10)
        result, _ = resample_to_biweekly(x, dates)
        assert result.shape == (24, 8, 8, 4)
        np.testing.assert_allclose(result, 0.42, atol=0.05)

    def test_various_input_counts(self):
        """Works with different numbers of input scenes."""
        for n_scenes in [5, 10, 15, 20, 30]:
            x = np.random.default_rng(42).uniform(0, 1, (n_scenes, 4, 4, 2)).astype(
                np.float32
            )
            dates = np.linspace(10, 350, n_scenes)
            result, max_gap = resample_to_biweekly(x, dates)
            assert result.shape == (24, 4, 4, 2)
            assert max_gap >= 0

    def test_output_spatial_shape_preserved(self, sample_s2_array):
        dates = np.linspace(0, 345, sample_s2_array.shape[0])
        result, _ = resample_to_biweekly(sample_s2_array, dates)
        assert result.shape[1:] == sample_s2_array.shape[1:]

    def test_returns_max_gap(self):
        x = np.random.default_rng(42).uniform(0, 1, (8, 4, 4, 2)).astype(np.float32)
        dates = np.array([10, 30, 50, 100, 150, 200, 300, 340], dtype=np.float64)
        _, max_gap = resample_to_biweekly(x, dates)
        assert isinstance(max_gap, int)
