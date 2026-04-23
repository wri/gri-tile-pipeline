"""Parity test: vegetation index formulas match the reference implementation."""

import numpy as np
import pytest

from gri_tile_pipeline.preprocessing.indices import bi, evi, grndvi, make_indices, msavi2

pytestmark = pytest.mark.parity


def _ref_evi(x):
    """Reference EVI: 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)."""
    blue = np.clip(x[..., 0], 0, 1)
    red = np.clip(x[..., 2], 0, 1)
    nir = np.clip(x[..., 3], 0, 1)
    return np.clip(2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1)), -1.5, 1.5)


def _ref_bi(x):
    """Reference BI: ((SWIR16+RED)-(NIR+BLUE)) / ((SWIR16+RED)+(NIR+BLUE))."""
    blue = np.clip(x[..., 0], 0, 1)
    red = np.clip(x[..., 2], 0, 1)
    nir = np.clip(x[..., 3], 0, 1)
    swir16 = np.clip(x[..., 8], 0, 1)
    return np.clip(
        ((swir16 + red) - (nir + blue)) / ((swir16 + red) + (nir + blue) + 1e-5),
        -1, 1,
    )


def _ref_msavi2(x):
    """Reference MSAVI2: (2*NIR+1 - sqrt((2*NIR+1)^2 - 8*(NIR-RED))) / 2."""
    red = np.clip(x[..., 2], 0, 1)
    nir = np.clip(x[..., 3], 0, 1)
    sqrt_arg = (2 * nir + 1) ** 2 - 8 * (nir - red)
    sqrt_arg[sqrt_arg < 0] = 0.0
    return np.clip((2 * nir + 1 - np.sqrt(sqrt_arg)) / 2, -1, 1)


def _ref_grndvi(x):
    """Reference GRNDVI: (NIR - (GREEN+RED)) / (NIR + (GREEN+RED))."""
    green = np.clip(x[..., 1], 0, 1)
    red = np.clip(x[..., 2], 0, 1)
    nir = np.clip(x[..., 3], 0, 1)
    return (nir - (green + red)) / (nir + green + red + 1e-5)


class TestEVIParity:
    def test_matches_reference(self, sample_s2_array):
        np.testing.assert_allclose(
            evi(sample_s2_array), _ref_evi(sample_s2_array), atol=1e-6
        )


class TestBIParity:
    def test_matches_reference(self, sample_s2_array):
        np.testing.assert_allclose(
            bi(sample_s2_array), _ref_bi(sample_s2_array), atol=1e-6
        )


class TestMSAVI2Parity:
    def test_matches_reference(self, sample_s2_array):
        np.testing.assert_allclose(
            msavi2(sample_s2_array), _ref_msavi2(sample_s2_array), atol=1e-6
        )


class TestGRNDVIParity:
    def test_matches_reference(self, sample_s2_array):
        np.testing.assert_allclose(
            grndvi(sample_s2_array), _ref_grndvi(sample_s2_array), atol=1e-6
        )
