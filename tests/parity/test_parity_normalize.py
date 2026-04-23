"""Parity test: normalization constants and formula match the reference."""

import numpy as np
import pytest

from gri_tile_pipeline.inference.normalize import MAX_ALL, MIN_ALL, normalize_subtile

pytestmark = pytest.mark.parity

# Reference constants from download_and_predict_job.py lines 1950-1963
# fmt: off
REF_MIN = np.array([
    0.006576638437476157, 0.0162050812542916, 0.010040436408026246,
    0.013351644159609368, 0.01965362020294499, 0.014229037918669413,
    0.015289539940489814, 0.011993591210803388, 0.008239871824216068,
    0.006546120393682765, 0.0, 0.0, 0.0, -0.1409399364817101,
    -0.4973397113668104, -0.09731556326714398, -0.7193834232943873,
], dtype=np.float32)

REF_MAX = np.array([
    0.2691233691920348, 0.3740291447318227, 0.5171435111009385,
    0.6027466239414053, 0.5650263218127718, 0.5747005416952773,
    0.5933928435187305, 0.6034943160143434, 0.7472037842374304,
    0.7000076295109483,
    0.4,
    0.948334642387533,
    0.6729257769285485, 0.8177635298774327, 0.35768999002433816,
    0.7545951919107605, 0.7602693339366691,
], dtype=np.float32)
# fmt: on


class TestNormalizationConstants:
    def test_min_all_matches_reference(self):
        np.testing.assert_allclose(MIN_ALL, REF_MIN, atol=1e-7)

    def test_max_all_matches_reference(self):
        np.testing.assert_allclose(MAX_ALL, REF_MAX, atol=1e-7)

    def test_17_channels(self):
        assert len(MIN_ALL) == 17
        assert len(MAX_ALL) == 17


class TestNormalizationFormula:
    def test_formula_parity(self):
        """Verify our normalization matches: (val - midrange) / (range/2)."""
        rng = np.random.default_rng(42)
        vals = rng.uniform(0, 0.5, (2, 4, 4, 17)).astype(np.float32)

        ours = normalize_subtile(vals)

        # Reference formula
        expected = vals.astype(np.float32, copy=True)
        for b in range(17):
            expected[..., b] = np.clip(expected[..., b], REF_MIN[b], REF_MAX[b])
            midrange = (REF_MAX[b] + REF_MIN[b]) / 2
            rng_b = REF_MAX[b] - REF_MIN[b]
            expected[..., b] = (expected[..., b] - midrange) / (rng_b / 2)

        np.testing.assert_allclose(ours, expected, atol=1e-5)
