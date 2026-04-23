"""Unit tests for preprocessing/whittaker.py."""

import numpy as np
import pytest

from gri_tile_pipeline.preprocessing.whittaker import WhittakerSmoother


class TestWhittakerSmoother:
    def test_output_shape_24_to_12(self):
        smoother = WhittakerSmoother(
            lmbd=100.0, size=24, nbands=10, dimx=8, dimy=8, outsize=12
        )
        x = np.random.default_rng(42).uniform(0, 1, (24, 8, 8, 10)).astype(np.float32)
        result = smoother.interpolate_array(x)
        assert result.shape == (12, 8, 8, 10)

    def test_smoothing_reduces_variance(self):
        rng = np.random.default_rng(42)
        noisy = rng.uniform(0, 1, (24, 8, 8, 4)).astype(np.float32)
        # Add sinusoidal signal + noise
        t = np.linspace(0, 2 * np.pi, 24)[:, None, None, None]
        signal = 0.5 + 0.3 * np.sin(t)
        noisy = (signal + 0.2 * noisy).astype(np.float32)

        smoother = WhittakerSmoother(
            lmbd=100.0, size=24, nbands=4, dimx=8, dimy=8, outsize=12
        )
        result = smoother.interpolate_array(noisy)

        # Smoothed output should have less variance across time
        assert np.var(result, axis=0).mean() < np.var(noisy, axis=0).mean()

    def test_higher_lambda_smoother(self):
        """Higher lambda should produce less pixel-level temporal variance in smoothed output."""
        rng = np.random.default_rng(42)
        # Use a signal with known structure: sinusoid + noise
        t = np.linspace(0, 4 * np.pi, 24)[:, None, None, None]
        base = 0.5 + 0.2 * np.sin(t)
        noise = rng.normal(0, 0.1, (24, 4, 4, 2))
        x = (base + noise).astype(np.float32)

        smoother_low = WhittakerSmoother(
            lmbd=1.0, size=24, nbands=2, dimx=4, dimy=4, outsize=24, average=False
        )
        smoother_high = WhittakerSmoother(
            lmbd=1e6, size=24, nbands=2, dimx=4, dimy=4, outsize=24, average=False
        )
        result_low = smoother_low.interpolate_array(x)
        result_high = smoother_high.interpolate_array(x)

        # Without averaging, high lambda should reduce temporal variation
        var_low = np.var(result_low, axis=0).mean()
        var_high = np.var(result_high, axis=0).mean()
        assert var_high < var_low

    def test_nbands_4_for_indices(self):
        """Supports nbands=4 for smoothing indices."""
        smoother = WhittakerSmoother(
            lmbd=100.0, size=24, nbands=4, dimx=16, dimy=16, outsize=12
        )
        x = np.random.default_rng(42).uniform(-0.5, 0.5, (24, 16, 16, 4)).astype(
            np.float32
        )
        result = smoother.interpolate_array(x)
        assert result.shape == (12, 16, 16, 4)

    def test_constant_input_passthrough(self):
        """Constant input should produce constant output."""
        x = np.full((24, 4, 4, 2), 0.5, dtype=np.float32)
        smoother = WhittakerSmoother(
            lmbd=100.0, size=24, nbands=2, dimx=4, dimy=4, outsize=12
        )
        result = smoother.interpolate_array(x)
        np.testing.assert_allclose(result, 0.5, atol=1e-4)
