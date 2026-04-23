"""Shared fixtures, markers, and skip conditions for the test suite."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

ARD_DIR = REPO_ROOT / "example" / "raw_v2"
REFERENCE_TIF = REPO_ROOT / "example" / "1000X871Y_FINAL.tif"
MODEL_DIR = REPO_ROOT / "models"

GOLDEN_DIR = REPO_ROOT / "example" / "golden"
GOLDEN_RAW = GOLDEN_DIR / "raw"
GOLDEN_TILES = ["1000X798Y", "1000X799Y", "1000X800Y"]

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

has_ard = pytest.mark.skipif(
    not ARD_DIR.is_dir(),
    reason=f"ARD directory not found: {ARD_DIR}",
)
has_reference = pytest.mark.skipif(
    not REFERENCE_TIF.is_file(),
    reason=f"Reference TIF not found: {REFERENCE_TIF}",
)
has_model = pytest.mark.skipif(
    not (MODEL_DIR / "predict_graph-172.pb").is_file(),
    reason=f"Model not found: {MODEL_DIR / 'predict_graph-172.pb'}",
)

_tf_available = False
try:
    import tensorflow  # noqa: F401
    _tf_available = True
except ImportError:
    pass

has_tf = pytest.mark.skipif(not _tf_available, reason="TensorFlow not installed")

has_golden = pytest.mark.skipif(
    not GOLDEN_DIR.is_dir(),
    reason=f"Golden test data not found: {GOLDEN_DIR}",
)

# ---------------------------------------------------------------------------
# Fixtures — paths
# ---------------------------------------------------------------------------


@pytest.fixture
def ard_dir() -> Path:
    if not ARD_DIR.is_dir():
        pytest.skip(f"ARD directory not found: {ARD_DIR}")
    return ARD_DIR


@pytest.fixture
def reference_tif() -> Path:
    if not REFERENCE_TIF.is_file():
        pytest.skip(f"Reference TIF not found: {REFERENCE_TIF}")
    return REFERENCE_TIF


@pytest.fixture
def model_dir() -> Path:
    model_file = MODEL_DIR / "predict_graph-172.pb"
    if not model_file.is_file():
        pytest.skip(f"Model not found: {model_file}")
    return MODEL_DIR


# ---------------------------------------------------------------------------
# Fixtures — synthetic arrays
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_s2_array() -> np.ndarray:
    """Synthetic S2 (T=8, H=32, W=32, B=10) float32 in [0, 1]."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.01, 0.5, (8, 32, 32, 10)).astype(np.float32)


@pytest.fixture
def sample_s1_array() -> np.ndarray:
    """Synthetic S1 (T=12, H=32, W=32, B=2) float32 in [0, 1]."""
    rng = np.random.default_rng(43)
    return rng.uniform(0.0, 1.0, (12, 32, 32, 2)).astype(np.float32)


@pytest.fixture
def sample_dem_array() -> np.ndarray:
    """Synthetic DEM (H=32, W=32) float32 in [0, 2000] meters."""
    rng = np.random.default_rng(44)
    return rng.uniform(0, 2000, (32, 32)).astype(np.float32)


@pytest.fixture
def sample_feature_stack() -> np.ndarray:
    """Synthetic feature stack (T=5, H=32, W=32, B=17) float32."""
    rng = np.random.default_rng(45)
    return rng.uniform(0, 0.5, (5, 32, 32, 17)).astype(np.float32)


@pytest.fixture
def sample_dates() -> np.ndarray:
    """24 evenly-spaced day-of-year values."""
    return np.linspace(0, 345, 24).astype(np.float64)
