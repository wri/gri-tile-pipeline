"""Unit tests for inference/normalize.py."""

import numpy as np
import pytest

from gri_tile_pipeline.inference.normalize import MAX_ALL, MIN_ALL, normalize_subtile


class TestConstants:
    def test_shape(self):
        assert MIN_ALL.shape == (17,)
        assert MAX_ALL.shape == (17,)

    def test_min_less_than_max(self):
        assert np.all(MIN_ALL < MAX_ALL)

    def test_dtype(self):
        assert MIN_ALL.dtype == np.float32
        assert MAX_ALL.dtype == np.float32

    def test_dem_channel(self):
        """Channel 10 (DEM): MIN=0, MAX=0.4."""
        assert MIN_ALL[10] == 0.0
        assert MAX_ALL[10] == pytest.approx(0.4, abs=1e-6)

    def test_s1_channels_start_zero(self):
        """Channels 11, 12 (S1 VV, VH): MIN=0."""
        assert MIN_ALL[11] == 0.0
        assert MIN_ALL[12] == 0.0


class TestNormalize:
    def test_output_range(self, sample_feature_stack):
        result = normalize_subtile(sample_feature_stack)
        assert result.min() >= -1.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_min_maps_to_minus_one(self):
        """Input equal to MIN_ALL should map to -1."""
        subtile = np.tile(MIN_ALL, (1, 1, 1, 1))
        result = normalize_subtile(subtile)
        np.testing.assert_allclose(result[0, 0, 0, :], -1.0, atol=1e-5)

    def test_max_maps_to_plus_one(self):
        """Input equal to MAX_ALL should map to +1."""
        subtile = np.tile(MAX_ALL, (1, 1, 1, 1))
        result = normalize_subtile(subtile)
        np.testing.assert_allclose(result[0, 0, 0, :], 1.0, atol=1e-5)

    def test_midpoint_maps_to_zero(self):
        """Input at the midpoint of MIN/MAX should map to 0."""
        midpoint = (MIN_ALL + MAX_ALL) / 2
        subtile = np.tile(midpoint, (1, 1, 1, 1))
        result = normalize_subtile(subtile)
        np.testing.assert_allclose(result[0, 0, 0, :], 0.0, atol=1e-5)

    def test_no_input_mutation(self, sample_feature_stack):
        original = sample_feature_stack.copy()
        _ = normalize_subtile(sample_feature_stack)
        np.testing.assert_array_equal(sample_feature_stack, original)

    def test_output_dtype(self, sample_feature_stack):
        result = normalize_subtile(sample_feature_stack)
        assert result.dtype == np.float32
