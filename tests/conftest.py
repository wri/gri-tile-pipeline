"""Shared fixtures, markers, and skip conditions for the test suite."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gri_shared_library.constants import TTC_TILEDB_FILEPATH, TERRAMATCH_GEOPARQUET_FILEPATH
from gri_shared_library.os_tools import remove_file
from gri_shared_library.productivity_tools import (download_ttc_test_data)
from gri_shared_library.geoparquet_tools import clear_ttc_for_test_projects, thin_tm_geoparquet_to_test_projects
from tests.constants import ARD_RAW, REFERENCE_TIF, MODEL_DIR, ARD_DIR, GOLDEN_DIR

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DOWNLOAD_GOLDEN_SCRIPT: Path = REPO_ROOT / "scripts" / "download_golden.py"

ARD_TEST_PREFIX = ["test_golden_", "test_ard_"]
VALID_MODES = ("include", "exclude", "only")


# ---------------------------------------------------------------------------
# Fixtures — paths
# ---------------------------------------------------------------------------

@pytest.fixture
def ard_dir() -> Path:
    os.makedirs(ARD_RAW, exist_ok=True)
    if not ARD_RAW.is_dir():
        pytest.skip(f"ARD directory not found: {ARD_RAW}")
    return ARD_RAW


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


def _download_ard_file_test_data() -> None:
    """Run scripts/download_golden.py to hydrate example/golden|raw_v2/."""
    subprocess.run([sys.executable, str(DOWNLOAD_GOLDEN_SCRIPT)], check=True)


def _setup_test_data(mode: str) -> None:
    if os.getenv("AWS_PROFILE"):
        # pre-execution cleanup
        remove_file(TTC_TILEDB_FILEPATH)
        remove_file(TERRAMATCH_GEOPARQUET_FILEPATH)
        # Prepare test projects
        download_ttc_test_data()
        if mode in ("include", "only"):
            _download_ard_file_test_data()
        thin_tm_geoparquet_to_test_projects()
        clear_ttc_for_test_projects()
        assert os.path.exists(TTC_TILEDB_FILEPATH)
        assert os.path.exists(TERRAMATCH_GEOPARQUET_FILEPATH)


def _remove_dir(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)


def _teardown_test_data(mode: str):
    if os.getenv("AWS_PROFILE"):
        # post-execution cleanup
        remove_file(TTC_TILEDB_FILEPATH)
        remove_file(TERRAMATCH_GEOPARQUET_FILEPATH)
        if mode in ("include", "only"):
            _remove_dir(ARD_DIR)
            _remove_dir(GOLDEN_DIR)
        assert not os.path.exists(TTC_TILEDB_FILEPATH)
        assert not os.path.exists(TERRAMATCH_GEOPARQUET_FILEPATH)


@pytest.hookimpl(tryfirst=True)  # must run before xdist's own pytest_configure
def pytest_configure(config):
    mode = _ard_file_mode(config)
    config._ard_file_mode = mode

    if _is_controller(config):
        # Runs on the controller before any worker is spawned -> setup happens once,
        # and the data files are on disk before the workers start
        # collecting/running.
        _setup_test_data(mode)

    if mode != "only":
        return  # only cap workers for the dedicated ard-file-only run

    if not hasattr(config.option, "numprocesses"):
        return  # pytest-xdist isn't installed/active - nothing to cap

    raw = config.getini("ard_file_max_workers")
    if raw:
        max_workers = int(raw)
    else:
        headroom = int(config.getini("ard_file_worker_headroom"))
        max_workers = max(1, (os.cpu_count() or 1) - headroom)

    requested = config.option.numprocesses  # None, 0, "auto", "logical", or an int
    try:
        new_value = min(int(requested), max_workers)
    except (TypeError, ValueError):
        # None, 0, "auto", "logical" all fall here -> force the capped value
        new_value = max_workers

    config.option.numprocesses = new_value


def pytest_unconfigure(config):
    # Runs on the controller after all workers have finished -> teardown once.
    if _is_controller(config):
        mode = _ard_file_mode(config)
        _teardown_test_data(mode)


def pytest_addoption(parser):
    parser.addini(
        "ard_file_tests",
        help="How to handle tests named 'test_golden_*' or 'test_ard_*': "
             "'include' (run everything, default), 'exclude' (skip ard-file tests), "
             "'only' (run only ard-file tests).",
        default="include",
    )
    parser.addini(
        "ard_file_max_workers",
        help="Absolute cap on xdist workers when ard_file_tests=only. "
             "Leave blank to derive it from ard_file_worker_headroom instead.",
        default="",
    )
    parser.addini(
        "ard_file_worker_headroom",
        help="When ard_file_max_workers is blank, use (vCPU count - this many) workers "
             "for ard-file-only runs. Default: 2.",
        default="2",
    )
    parser.addoption(
        "--ard-file",
        choices=VALID_MODES,
        default=None,
        help="Override the ard_file_tests ini setting for this run.",
    )


def _ard_file_mode(config):
    mode = config.getoption("ard_file") or config.getini("ard_file_tests")
    if mode not in VALID_MODES:
        raise pytest.UsageError(
            f"invalid ard_file_tests value {mode!r}; must be one of {VALID_MODES}"
        )
    return mode


def pytest_collection_modifyitems(config, items):
    mode = getattr(config, "_ard_file_mode", None) or _ard_file_mode(config)

    if mode == "include":
        return

    if mode == "exclude":
        marker = pytest.mark.skip(reason="excluded: ard_file_tests=exclude")
        target = lambda name: any(name.startswith(prefix) for prefix in ARD_TEST_PREFIX)
    else:  # mode == "only"
        marker = pytest.mark.skip(reason="skipped: ard_file_tests=only")
        target = lambda name: not any(name.startswith(prefix) for prefix in ARD_TEST_PREFIX)

    for item in items:
        if target(item.name):
            item.add_marker(marker)
