"""Unit tests for loaders/shared.py."""

import sys
import os

import numpy as np
import pytest

# Add loaders to path so we can import shared directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "loaders"))

from loaders.shared import compute_band_stats, make_bbox


class TestMakeBbox:
    def test_expansion(self):
        bbx = [10.0, 20.0, 10.0, 20.0]
        result = make_bbox(bbx, expansion=10)
        multiplier = 1 / 360
        assert result[0] == pytest.approx(10.0 - 10 * multiplier)
        assert result[1] == pytest.approx(20.0 - 10 * multiplier)
        assert result[2] == pytest.approx(10.0 + 10 * multiplier)
        assert result[3] == pytest.approx(20.0 + 10 * multiplier)

    def test_no_mutation(self):
        bbx = [10.0, 20.0, 10.0, 20.0]
        original = bbx.copy()
        _ = make_bbox(bbx)
        assert bbx == original

    def test_zero_expansion(self):
        bbx = [10.0, 20.0, 10.0, 20.0]
        result = make_bbox(bbx, expansion=0)
        assert result == bbx


class TestComputeBandStats:
    def test_valid_array(self):
        arr = np.array([100, 200, 300, 400, 500], dtype=np.uint16)
        stats = compute_band_stats(arr)
        assert stats["min"] == 100
        assert stats["max"] == 500
        assert stats["mean"] == pytest.approx(300.0)
        assert stats["valid_count"] == 5
        assert stats["valid_ratio"] == pytest.approx(1.0)

    def test_zero_handling(self):
        arr = np.array([0, 0, 0, 100, 200], dtype=np.uint16)
        stats = compute_band_stats(arr)
        assert stats["valid_count"] == 2
        assert stats["min"] == 100
        assert stats["valid_ratio"] == pytest.approx(0.4)

    def test_all_zeros(self):
        arr = np.zeros(10, dtype=np.uint16)
        stats = compute_band_stats(arr)
        assert stats["valid_count"] == 0
        assert stats["mean"] == 0.0
        assert stats["valid_ratio"] == 0.0
        assert stats["count"] == 10

    def test_percentiles(self):
        arr = np.arange(1, 101, dtype=np.uint16)
        stats = compute_band_stats(arr)
        assert stats["p5"] == pytest.approx(5.95, abs=1)
        assert stats["p50"] == pytest.approx(50.5, abs=1)
        assert stats["p95"] == pytest.approx(95.05, abs=1)
