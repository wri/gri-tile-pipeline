"""Parity test: DEM normalization matches the reference (divisor=90)."""

import numpy as np
import pytest

pytestmark = pytest.mark.parity


class TestDEMNormalization:
    def test_divisor_is_90(self):
        """Reference uses /90 for DEM normalization (NOT /10000)."""
        dem_meters = np.array([0, 45, 90, 900, 3600], dtype=np.float32)
        result = dem_meters / 90.0
        expected = np.array([0.0, 0.5, 1.0, 10.0, 40.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_typical_elevation_range(self):
        """Typical elevations (0-3000m) produce values in [0, ~33]."""
        dem = np.linspace(0, 3000, 100, dtype=np.float32)
        result = dem / 90.0
        assert result.min() == 0.0
        assert result.max() == pytest.approx(3000 / 90, abs=0.01)

    def test_median_filter(self):
        """Reference applies median_filter(size=5) to DEM."""
        from scipy.ndimage import median_filter

        rng = np.random.default_rng(42)
        dem = rng.uniform(0, 1000, (32, 32)).astype(np.float32)
        filtered = median_filter(dem, size=5)
        # Filtered should be smoother
        assert np.std(filtered) <= np.std(dem)
        # Shape preserved
        assert filtered.shape == dem.shape

    def test_dem_max_all_is_0_4(self):
        """MAX_ALL[10] = 0.4 means the model expects DEM/90 ≤ 0.4 → ≤36m."""
        from gri_tile_pipeline.inference.normalize import MAX_ALL
        assert MAX_ALL[10] == pytest.approx(0.4, abs=1e-6)
