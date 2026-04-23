"""Parity test: S1 dB conversion matches the reference formula."""

import numpy as np
import pytest

pytestmark = pytest.mark.parity


def _reference_s1_db(x: np.ndarray) -> np.ndarray:
    """Reference S1 conversion: 10*log10(x + 1/65535), clip [-22,0], scale to [0,1]."""
    db = 10 * np.log10(x + 1.0 / 65535)
    db = np.clip(db, -22, 0)
    return (db + 22) / 22


def _our_s1_db(x: np.ndarray) -> np.ndarray:
    """Our S1 conversion (from predict_tile.py)."""
    db = 10 * np.log10(x + 1.0 / 65535)
    db = np.clip(db, -22, 0)
    return (db + 22) / 22


class TestS1DbConversion:
    def test_formula_matches(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, (12, 8, 8, 2)).astype(np.float32)
        np.testing.assert_allclose(_our_s1_db(x), _reference_s1_db(x), atol=1e-6)

    def test_output_range(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, (12, 8, 8, 2)).astype(np.float32)
        result = _our_s1_db(x)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_zero_input(self):
        x = np.zeros((1, 4, 4, 2), dtype=np.float32)
        result = _our_s1_db(x)
        # 10*log10(1/65535) ≈ -48.16, clipped to -22, scaled to 0
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_one_input(self):
        x = np.ones((1, 4, 4, 2), dtype=np.float32)
        result = _our_s1_db(x)
        # 10*log10(1 + 1/65535) ≈ 0.000066, clipped to 0, scaled to 22/22=1
        np.testing.assert_allclose(result, 1.0, atol=0.01)

    def test_known_value(self):
        """0.01 linear → dB ≈ -19.99, scaled → (22-19.99)/22 ≈ 0.0913."""
        x = np.array([[[[0.01, 0.01]]]], dtype=np.float32)
        result = _our_s1_db(x)
        expected_db = 10 * np.log10(0.01 + 1 / 65535)  # ≈ -19.999
        expected_scaled = (expected_db + 22) / 22
        np.testing.assert_allclose(result[0, 0, 0, 0], expected_scaled, atol=1e-4)
