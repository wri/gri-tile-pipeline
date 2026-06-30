"""Shared fixtures, markers, and skip conditions for the test suite."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gri_shared_library.constants import TTC_TILEDB_FILEPATH, TERRAMATCH_GEOPARQUET_FILEPATH
from gri_shared_library.os_tools import remove_file
from gri_shared_library.productivity_tools import (download_ttc_test_data)
from gri_shared_library.geoparquet_tools import clear_ttc_for_test_projects, thin_tm_geoparquet_to_test_projects
from tests.constants import ARD_DIR, REFERENCE_TIF, MODEL_DIR

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


def _is_controller(config):
    # pytest-xdist sets `workerinput` on each worker process. The controller
    # (and any non-distributed run, e.g. without -n) does NOT have it, so this
    # is True exactly once per test session, regardless of worker count.
    return not hasattr(config, "workerinput")


def _setup_test_data():
    # pre-execution cleanup
    remove_file(TTC_TILEDB_FILEPATH)
    remove_file(TERRAMATCH_GEOPARQUET_FILEPATH)
    # Prepare test projects
    download_ttc_test_data()
    thin_tm_geoparquet_to_test_projects()
    clear_ttc_for_test_projects()
    assert os.path.exists(TTC_TILEDB_FILEPATH)
    assert os.path.exists(TERRAMATCH_GEOPARQUET_FILEPATH)


def _teardown_test_data():
    # post-execution cleanup
    remove_file(TTC_TILEDB_FILEPATH)
    remove_file(TERRAMATCH_GEOPARQUET_FILEPATH)
    assert not os.path.exists(TTC_TILEDB_FILEPATH)
    assert not os.path.exists(TERRAMATCH_GEOPARQUET_FILEPATH)


def pytest_configure(config):
    # Runs on the controller before any worker is spawned -> setup happens once,
    # and the data files are on disk before the workers start
    # collecting/running.
    if _is_controller(config):
        _setup_test_data()


def pytest_unconfigure(config):
    # Runs on the controller after all workers have finished -> teardown once.
    if _is_controller(config):
        _teardown_test_data()
