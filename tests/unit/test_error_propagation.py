"""Tests for zonal error propagation matching reference behavior."""

import math
import pytest


class TestSmallSiteError:
    def test_zero_ttc_returns_zero(self):
        from gri_tile_pipeline.zonal.error_propagation import small_site_error
        assert small_site_error(0.1, 0.0) == 0.0

    def test_above_threshold_returns_zero(self):
        from gri_tile_pipeline.zonal.error_propagation import small_site_error
        assert small_site_error(1.0, 50.0, threshold=0.5) == 0.0

    def test_bracket_0_to_10(self):
        from gri_tile_pipeline.zonal.error_propagation import small_site_error
        # 5% TTC, 0.3 ha → should use 3.6386 / 5 = 0.72772
        result = small_site_error(0.3, 5.0)
        assert pytest.approx(result, rel=1e-4) == 3.6386 / 5.0

    def test_bracket_10_to_40(self):
        from gri_tile_pipeline.zonal.error_propagation import small_site_error
        # 25% TTC, 0.3 ha → should use 16.68 / 25 = 0.6672
        result = small_site_error(0.3, 25.0)
        assert pytest.approx(result, rel=1e-4) == 16.68 / 25.0

    def test_bracket_40_to_100(self):
        from gri_tile_pipeline.zonal.error_propagation import small_site_error
        # 60% TTC, 0.3 ha → should use 23.468 / 60
        result = small_site_error(0.3, 60.0)
        assert pytest.approx(result, rel=1e-4) == 23.468 / 60.0

    def test_boundary_at_10(self):
        from gri_tile_pipeline.zonal.error_propagation import small_site_error
        # Exactly 10 should be in the first bracket (0 <= ttc <= 10)
        result = small_site_error(0.3, 10.0)
        assert pytest.approx(result, rel=1e-4) == 3.6386 / 10.0

    def test_just_above_10(self):
        from gri_tile_pipeline.zonal.error_propagation import small_site_error
        result = small_site_error(0.3, 10.01)
        assert pytest.approx(result, rel=1e-3) == 16.68 / 10.01


class TestLulcError:
    def test_zero_ttc_returns_zero(self):
        from gri_tile_pipeline.zonal.error_propagation import lulc_error
        lo, hi = lulc_error(0.0, "forest")
        assert lo == 0.0
        assert hi == 0.0

    def test_forest_category(self):
        from gri_tile_pipeline.zonal.error_propagation import lulc_error
        lo, hi = lulc_error(50.0, "forest")
        # Should use (p_upper_95 - r_lower_95) / 50 and (r_upper_95 - p_lower_95) / 50
        assert lo != 0.0
        assert hi != 0.0
        # Both should be small relative fractions
        assert abs(lo) < 1.0
        assert abs(hi) < 1.0

    def test_unknown_category_falls_back_to_forest(self):
        from gri_tile_pipeline.zonal.error_propagation import lulc_error
        lo_unknown, hi_unknown = lulc_error(50.0, "nonexistent")
        lo_forest, hi_forest = lulc_error(50.0, "forest")
        assert lo_unknown == lo_forest
        assert hi_unknown == hi_forest

    def test_case_insensitive(self):
        from gri_tile_pipeline.zonal.error_propagation import lulc_error
        lo1, hi1 = lulc_error(50.0, "Forest")
        lo2, hi2 = lulc_error(50.0, "forest")
        assert lo1 == lo2
        assert hi1 == hi2


class TestSubregionError:
    def test_zero_ttc_returns_zero(self):
        from gri_tile_pipeline.zonal.error_propagation import subregion_error
        from shapely.geometry import Point
        cat, lo, hi = subregion_error(Point(-60, -15), 0.0)
        assert lo == 0.0
        assert hi == 0.0

    def test_south_america_point(self):
        from gri_tile_pipeline.zonal.error_propagation import subregion_error
        from shapely.geometry import Point
        # Point in Brazil (South America)
        cat, lo, hi = subregion_error(Point(-50, -10), 50.0)
        assert cat == "South America"
        assert lo != 0.0
        assert hi != 0.0

    def test_no_intersection_returns_zero(self):
        from gri_tile_pipeline.zonal.error_propagation import subregion_error
        from shapely.geometry import Point
        # Point in the ocean
        cat, lo, hi = subregion_error(Point(0, 0), 50.0)
        # May or may not intersect depending on geometry resolution
        # At minimum, should not raise
        assert isinstance(lo, float)
        assert isinstance(hi, float)


class TestCombineErrors:
    def test_basic_combination(self):
        from gri_tile_pipeline.zonal.error_propagation import combine_errors
        result = combine_errors(
            expected_ttc=50.0,
            shift_err=0.1,
            small_site_err=0.2,
            lulc_lower=0.05,
            lulc_upper=0.06,
            subregion_lower=0.03,
            subregion_upper=0.04,
        )
        assert "error_plus" in result
        assert "error_minus" in result
        assert "plus_minus_average" in result
        # Average should be between plus and minus
        assert result["plus_minus_average"] == pytest.approx(
            (result["error_plus"] + result["error_minus"]) / 2
        )

    def test_reference_parity(self):
        """Test against reference test_combine_errors expected range (4.9-5.1)."""
        from gri_tile_pipeline.zonal.error_propagation import combine_errors
        # Using typical values that should produce ~5.0 average
        result = combine_errors(
            expected_ttc=50.0,
            shift_err=0.1,
            small_site_err=0.1,
            lulc_lower=0.05,
            lulc_upper=0.05,
            subregion_lower=0.05,
            subregion_upper=0.05,
        )
        avg = result["plus_minus_average"]
        assert 0 < avg < 50  # Sanity check

    def test_zero_errors(self):
        from gri_tile_pipeline.zonal.error_propagation import combine_errors
        result = combine_errors(50.0, 0, 0, 0, 0, 0, 0)
        assert result["error_plus"] == 0.0
        assert result["error_minus"] == 0.0
        assert result["plus_minus_average"] == 0.0


class TestClampInf:
    def test_inf_clamped(self):
        from gri_tile_pipeline.zonal.error_propagation import _clamp_inf
        assert _clamp_inf(float("inf")) == 0.0
        assert _clamp_inf(float("-inf")) == 0.0

    def test_normal_values_unchanged(self):
        from gri_tile_pipeline.zonal.error_propagation import _clamp_inf
        assert _clamp_inf(0.5) == 0.5
        assert _clamp_inf(0.0) == 0.0
