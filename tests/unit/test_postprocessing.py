"""Unit tests for inference/postprocessing.py."""

import numpy as np
import pytest

from gri_tile_pipeline.inference.postprocessing import apply_nodata_mask


class TestApplyNodataMask:
    def test_zeros_become_255(self):
        """Pixels with all-zero S2 data across time should become nodata (255)."""
        preds = np.full((32, 32), 50, dtype=np.uint8)
        s2 = np.random.default_rng(42).uniform(0.01, 0.5, (5, 32, 32, 10)).astype(
            np.float32
        )
        # Make one column all-zero in first 4 bands across all timesteps
        s2[:, :, 0, :4] = 0.0

        result = apply_nodata_mask(preds, s2)
        # The all-zero column (plus dilation) should be 255
        assert np.any(result == 255)

    def test_valid_data_preserved(self):
        """Non-zero S2 pixels should keep their predictions."""
        preds = np.full((32, 32), 42, dtype=np.uint8)
        s2 = np.random.default_rng(42).uniform(0.05, 0.5, (5, 32, 32, 10)).astype(
            np.float32
        )
        result = apply_nodata_mask(preds, s2)
        # Interior pixels (away from any potential edge dilation) should be preserved
        assert np.all(result[10:20, 10:20] == 42)

    def test_nan_detected(self):
        """NaN values should trigger nodata masking."""
        preds = np.full((32, 32), 50, dtype=np.uint8)
        s2 = np.random.default_rng(42).uniform(0.05, 0.5, (5, 32, 32, 10)).astype(
            np.float32
        )
        s2[0, 16, 16, 0] = np.nan
        result = apply_nodata_mask(preds, s2)
        # NaN pixel + dilation should be masked
        assert result[16, 16] == 255

    def test_output_shape(self):
        preds = np.full((32, 32), 50, dtype=np.uint8)
        s2 = np.random.default_rng(42).uniform(0.05, 0.5, (5, 32, 32, 10)).astype(
            np.float32
        )
        result = apply_nodata_mask(preds, s2)
        assert result.shape == preds.shape
        assert result.dtype == np.uint8
