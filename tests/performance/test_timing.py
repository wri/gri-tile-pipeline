"""Performance benchmarks: ensure key operations meet timing targets."""

import time

import numpy as np
import pytest

pytestmark = pytest.mark.slow


class TestWhittakerTiming:
    def test_whittaker_large(self):
        """Whittaker smoothing on (24, 618, 616, 10) should complete in <10s."""
        from gri_tile_pipeline.preprocessing.whittaker import WhittakerSmoother

        H, W = 618, 616
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, (24, H, W, 10)).astype(np.float32)

        smoother = WhittakerSmoother(
            lmbd=100.0, size=24, nbands=10, dimx=H, dimy=W, outsize=12
        )

        start = time.perf_counter()
        result = smoother.interpolate_array(x)
        elapsed = time.perf_counter() - start

        assert result.shape == (12, H, W, 10)
        assert elapsed < 10, f"Whittaker took {elapsed:.1f}s (target <10s)"
        print(f"  Whittaker (618x616x10): {elapsed:.2f}s")


class TestIndexTiming:
    def test_index_computation(self):
        """Index computation on (24, 618, 616, 10) should complete in <2s."""
        from gri_tile_pipeline.preprocessing.indices import make_indices

        rng = np.random.default_rng(42)
        x = rng.uniform(0.01, 0.5, (24, 618, 616, 10)).astype(np.float32)

        start = time.perf_counter()
        result = make_indices(x)
        elapsed = time.perf_counter() - start

        assert result.shape == (24, 618, 616, 4)
        assert elapsed < 2, f"Indices took {elapsed:.1f}s (target <2s)"
        print(f"  Indices (24x618x616): {elapsed:.2f}s")


class TestNormalizeTiming:
    def test_normalize_subtile(self):
        """Normalization on (5, 172, 172, 17) should complete in <500ms."""
        from gri_tile_pipeline.inference.normalize import normalize_subtile

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 0.5, (5, 172, 172, 17)).astype(np.float32)

        start = time.perf_counter()
        result = normalize_subtile(x)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"Normalization took {elapsed:.3f}s (target <0.5s)"
        print(f"  Normalize (5x172x172x17): {elapsed*1000:.1f}ms")


class TestTemporalResamplingTiming:
    def test_resample_biweekly(self):
        """Temporal resampling on (15, 618, 616, 10) should complete in <5s."""
        from gri_tile_pipeline.preprocessing.temporal_resampling import (
            resample_to_biweekly,
        )

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, (15, 618, 616, 10)).astype(np.float32)
        dates = np.linspace(10, 350, 15)

        start = time.perf_counter()
        result, _ = resample_to_biweekly(x, dates)
        elapsed = time.perf_counter() - start

        assert result.shape[0] == 24
        assert elapsed < 5, f"Resampling took {elapsed:.1f}s (target <5s)"
        print(f"  Temporal resampling (15x618x616x10): {elapsed:.2f}s")


class TestMemoryEstimate:
    def test_feature_stack_memory(self):
        """Feature stack (5, 618, 616, 17) should use <200 MB."""
        shape = (5, 618, 616, 17)
        bytes_needed = np.prod(shape) * 4  # float32
        mb_needed = bytes_needed / (1024 * 1024)
        assert mb_needed < 200, f"Feature stack needs {mb_needed:.1f} MB (target <200 MB)"
        print(f"  Feature stack memory: {mb_needed:.1f} MB")
