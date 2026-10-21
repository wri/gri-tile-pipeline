"""Unit tests for inference/subtile_predict.py."""

import numpy as np
import pytest

from gri_tile_pipeline.inference.subtile_predict import (
    BORDER,
    INPUT_SIZE,
    SIZE,
    fspecial_gauss,
    make_windows,
)


class TestConstants:
    def test_size(self):
        assert SIZE == 158

    def test_border(self):
        assert BORDER == 7

    def test_input_size(self):
        assert INPUT_SIZE == SIZE + 2 * BORDER
        assert INPUT_SIZE == 172


class TestFspecialGauss:
    def test_shape(self):
        g = fspecial_gauss(158, 36)
        assert g.shape == (158, 158)

    def test_symmetry_odd(self):
        """Odd-sized kernel should be perfectly symmetric."""
        g = fspecial_gauss(157, 36)
        np.testing.assert_allclose(g, g[::-1, :], atol=1e-10)
        np.testing.assert_allclose(g, g[:, ::-1], atol=1e-10)

    def test_approximate_symmetry_even(self):
        """Even-sized kernel (158) has near-symmetric Gaussian profile."""
        g = fspecial_gauss(158, 36)
        # For even sizes, mgrid produces asymmetric range (-78..79),
        # so only approximate symmetry is expected.
        assert g[0, 79] == pytest.approx(g[0, 78], rel=0.2)

    def test_peak_at_center(self):
        g = fspecial_gauss(158, 36)
        # Peak should be near center
        peak_idx = np.unravel_index(g.argmax(), g.shape)
        center = 158 // 2
        assert abs(peak_idx[0] - center) <= 1
        assert abs(peak_idx[1] - center) <= 1

    def test_positive(self):
        g = fspecial_gauss(158, 36)
        assert np.all(g > 0)

    def test_small_kernel(self):
        g = fspecial_gauss(3, 1)
        assert g.shape == (3, 3)
        assert g[1, 1] == g.max()


class TestMakeWindows:
    def test_full_coverage(self):
        """Windows should cover the full spatial extent."""
        H, W = 618, 616
        tiles_folder, tiles_array = make_windows(H, W)

        # Check that every pixel is covered by at least one window
        coverage = np.zeros((H, W), dtype=bool)
        for i in range(len(tiles_folder)):
            x, y, w, h = tiles_folder[i]
            coverage[x : x + w, y : y + h] = True
        assert coverage.all()

    def test_window_count(self):
        """Should produce a reasonable number of windows."""
        H, W = 618, 616
        tiles_folder, tiles_array = make_windows(H, W)
        assert len(tiles_folder) > 0
        assert len(tiles_folder) == len(tiles_array)

    def test_tiles_array_expanded(self):
        """tiles_array should be larger than tiles_folder by BORDER."""
        H, W = 618, 616
        tiles_folder, tiles_array = make_windows(H, W)
        # Interior windows should be expanded by BORDER
        for i in range(len(tiles_folder)):
            assert tiles_array[i, 2] >= tiles_folder[i, 2]
            assert tiles_array[i, 3] >= tiles_folder[i, 3]

    def test_image_larger_than_tile(self):
        """Should work for images moderately larger than tile_size."""
        H, W = 200, 200
        tiles_folder, tiles_array = make_windows(H, W, tile_size=SIZE)
        assert len(tiles_folder) >= 1

    def test_array_clipped_to_bounds(self):
        """tiles_array should not exceed spatial bounds."""
        H, W = 618, 616
        _, tiles_array = make_windows(H, W)
        for i in range(len(tiles_array)):
            x, y, w, h = tiles_array[i]
            assert x >= 0
            assert y >= 0
            assert x + w <= H
            assert y + h <= W
