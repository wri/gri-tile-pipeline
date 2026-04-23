"""Regression tests: compare deterministic computations against golden .npy files.

On first run, golden files are created. On subsequent runs, output must match.
"""

import os

import numpy as np
import pytest

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


def _golden_path(name: str) -> str:
    return os.path.join(GOLDEN_DIR, name)


def _check_or_create_golden(result: np.ndarray, name: str, atol: float = 1e-5):
    """Compare against golden file, or create it if it doesn't exist."""
    path = _golden_path(name)
    if os.path.exists(path):
        expected = np.load(path)
        np.testing.assert_allclose(
            result, expected, atol=atol,
            err_msg=f"Golden file mismatch: {name}",
        )
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, result)
        pytest.skip(f"Golden file created: {name} (run again to validate)")


class TestGoldenIndices:
    def test_make_indices(self):
        from gri_tile_pipeline.preprocessing.indices import make_indices

        rng = np.random.default_rng(100)
        x = rng.uniform(0.01, 0.5, (4, 16, 16, 10)).astype(np.float32)
        result = make_indices(x)
        _check_or_create_golden(result, "indices_4x16x16.npy")


class TestGoldenNormalize:
    def test_normalize_subtile(self):
        from gri_tile_pipeline.inference.normalize import normalize_subtile

        rng = np.random.default_rng(101)
        x = rng.uniform(0, 0.5, (5, 8, 8, 17)).astype(np.float32)
        result = normalize_subtile(x)
        _check_or_create_golden(result, "normalize_5x8x8.npy")


class TestGoldenWhittaker:
    def test_whittaker_24to12(self):
        from gri_tile_pipeline.preprocessing.whittaker import WhittakerSmoother

        rng = np.random.default_rng(102)
        x = rng.uniform(0, 1, (24, 8, 8, 4)).astype(np.float32)
        smoother = WhittakerSmoother(
            lmbd=100.0, size=24, nbands=4, dimx=8, dimy=8, outsize=12
        )
        result = smoother.interpolate_array(x)
        _check_or_create_golden(result, "whittaker_24to12_8x8.npy")


class TestGoldenGaussian:
    def test_gauss_kernel(self):
        from gri_tile_pipeline.inference.subtile_predict import fspecial_gauss

        result = fspecial_gauss(158, 36)
        _check_or_create_golden(result, "gauss_158_36.npy")
