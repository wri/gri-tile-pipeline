"""Unit tests for preprocessing/super_resolution.py."""

import numpy as np
import pytest

from gri_tile_pipeline.preprocessing.super_resolution import superresolve_tile


class TestSuperresolveTile:
    def test_no_session_passthrough(self):
        """Without a TF session, input is returned unchanged."""
        x = np.random.default_rng(42).uniform(0, 1, (4, 64, 64, 10)).astype(
            np.float32
        )
        original = x.copy()
        result = superresolve_tile(x, sess=None)
        np.testing.assert_array_equal(result, original)

    def test_shape_preserved(self):
        x = np.random.default_rng(42).uniform(0, 1, (4, 64, 64, 10)).astype(
            np.float32
        )
        result = superresolve_tile(x, sess=None)
        assert result.shape == x.shape

    def test_dtype_preserved(self):
        x = np.random.default_rng(42).uniform(0, 1, (4, 64, 64, 10)).astype(
            np.float32
        )
        result = superresolve_tile(x, sess=None)
        assert result.dtype == np.float32
