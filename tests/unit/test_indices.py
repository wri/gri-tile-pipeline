"""Unit tests for preprocessing/indices.py."""

import numpy as np
import pytest

from gri_tile_pipeline.preprocessing.indices import bi, evi, grndvi, make_indices, msavi2


@pytest.fixture
def vegetation_pixel():
    """Typical vegetation: high NIR, low RED/BLUE."""
    # (1, 1, 1, 10) — one pixel, 10 bands
    x = np.zeros((1, 1, 1, 10), dtype=np.float32)
    x[..., 0] = 0.03  # Blue
    x[..., 1] = 0.05  # Green
    x[..., 2] = 0.04  # Red
    x[..., 3] = 0.40  # NIR
    x[..., 8] = 0.10  # SWIR16
    return x


@pytest.fixture
def bare_soil_pixel():
    """Bare soil: high visible, low NIR."""
    x = np.zeros((1, 1, 1, 10), dtype=np.float32)
    x[..., 0] = 0.15  # Blue
    x[..., 1] = 0.18  # Green
    x[..., 2] = 0.20  # Red
    x[..., 3] = 0.22  # NIR
    x[..., 8] = 0.30  # SWIR16
    return x


class TestEVI:
    def test_vegetation_positive(self, vegetation_pixel):
        result = evi(vegetation_pixel)
        assert result.item() > 0.3

    def test_bare_soil_low(self, bare_soil_pixel):
        result = evi(bare_soil_pixel)
        assert result.item() < 0.3

    def test_range(self, sample_s2_array):
        result = evi(sample_s2_array)
        assert result.min() >= -1.5
        assert result.max() <= 1.5

    def test_division_by_zero_safe(self):
        x = np.zeros((1, 1, 1, 10), dtype=np.float32)
        result = evi(x)
        assert np.isfinite(result).all()


class TestBI:
    def test_bare_soil_positive(self, bare_soil_pixel):
        result = bi(bare_soil_pixel)
        assert result.item() > 0

    def test_vegetation_negative(self, vegetation_pixel):
        result = bi(vegetation_pixel)
        assert result.item() < 0

    def test_range(self, sample_s2_array):
        result = bi(sample_s2_array)
        assert result.min() >= -1
        assert result.max() <= 1


class TestMSAVI2:
    def test_vegetation_high(self, vegetation_pixel):
        result = msavi2(vegetation_pixel)
        assert result.item() > 0.2

    def test_range(self, sample_s2_array):
        result = msavi2(sample_s2_array)
        assert result.min() >= -1
        assert result.max() <= 1

    def test_negative_sqrt_arg_safe(self):
        """sqrt argument clipped to 0 when negative."""
        x = np.zeros((1, 1, 1, 10), dtype=np.float32)
        x[..., 2] = 1.0  # Red very high
        x[..., 3] = 0.0  # NIR zero
        result = msavi2(x)
        assert np.isfinite(result).all()


class TestGRNDVI:
    def test_vegetation_positive(self, vegetation_pixel):
        result = grndvi(vegetation_pixel)
        assert result.item() > 0

    def test_division_by_zero_safe(self):
        x = np.zeros((1, 1, 1, 10), dtype=np.float32)
        result = grndvi(x)
        assert np.isfinite(result).all()


class TestMakeIndices:
    def test_output_shape(self, sample_s2_array):
        result = make_indices(sample_s2_array)
        assert result.shape == (8, 32, 32, 4)

    def test_output_dtype(self, sample_s2_array):
        result = make_indices(sample_s2_array)
        assert result.dtype == np.float32

    def test_channel_order(self, vegetation_pixel):
        """Channel order: [EVI, BI, MSAVI2, GRNDVI]."""
        result = make_indices(vegetation_pixel)
        assert result[..., 0].item() == pytest.approx(evi(vegetation_pixel).item(), abs=1e-6)
        assert result[..., 1].item() == pytest.approx(bi(vegetation_pixel).item(), abs=1e-6)
        assert result[..., 2].item() == pytest.approx(msavi2(vegetation_pixel).item(), abs=1e-6)
        assert result[..., 3].item() == pytest.approx(grndvi(vegetation_pixel).item(), abs=1e-6)
