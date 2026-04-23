"""Golden tile parity tests: run prediction on 3 golden tiles and compare to reference TIFs.

Usage:
    pytest tests/parity/test_golden_parity.py -v -s
    pytest tests/parity/test_golden_parity.py -v -s -k "baseline"
    pytest tests/parity/test_golden_parity.py -v -s -k "improved"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import GOLDEN_DIR, GOLDEN_RAW, GOLDEN_TILES, MODEL_DIR
from tests.parity.metrics import (
    aggregate_golden_report,
    compare_predictions,
    print_parity_table,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Ensure loaders/ is importable
_loader_path = str(REPO_ROOT / "loaders")
if _loader_path not in sys.path:
    sys.path.insert(0, _loader_path)

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_golden_exists = GOLDEN_DIR.is_dir() and all(
    (GOLDEN_DIR / f"{t}_FINAL.tif").is_file() for t in GOLDEN_TILES
)
_model_exists = (MODEL_DIR / "predict_graph-172.pb").is_file()

_tf_available = False
try:
    import tensorflow  # noqa: F401

    _tf_available = True
except ImportError:
    pass

pytestmark = [
    pytest.mark.parity,
    pytest.mark.slow,
    pytest.mark.tf,
    pytest.mark.skipif(not _golden_exists, reason="Golden test data not found"),
    pytest.mark.skipif(not _model_exists, reason="Model not found"),
    pytest.mark.skipif(not _tf_available, reason="TensorFlow not installed"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_golden_tile(tile_name: str) -> dict:
    """Load all ARD arrays for a specific golden tile by name."""
    import hickle as hkl

    result = {
        "s2_10": hkl.load(str(GOLDEN_RAW / "s2_10" / f"{tile_name}.hkl")),
        "s2_20": hkl.load(str(GOLDEN_RAW / "s2_20" / f"{tile_name}.hkl")),
        "s1": hkl.load(str(GOLDEN_RAW / "s1" / f"{tile_name}.hkl")),
        "dem": hkl.load(str(GOLDEN_RAW / "misc" / f"dem_{tile_name}.hkl")),
        "clouds": hkl.load(str(GOLDEN_RAW / "clouds" / f"clouds_{tile_name}.hkl")),
        "s2_dates": hkl.load(
            str(GOLDEN_RAW / "misc" / f"s2_dates_{tile_name}.hkl")
        ),
    }
    clm_path = GOLDEN_RAW / "clouds" / f"cloudmask_{tile_name}.hkl"
    if clm_path.exists():
        result["clm"] = hkl.load(str(clm_path))
    return result


def load_reference_tif(tile_name: str) -> np.ndarray:
    """Load the golden reference TIF for a tile."""
    import rasterio

    with rasterio.open(str(GOLDEN_DIR / f"{tile_name}_FINAL.tif")) as src:
        return src.read(1)


# ---------------------------------------------------------------------------
# Session-scoped prediction cache
# ---------------------------------------------------------------------------

# Store predictions so they're computed once per session, not per test.
_prediction_cache: dict[str, np.ndarray] = {}
_stats_cache: dict[str, dict] = {}


def _get_prediction(tile_name: str) -> np.ndarray:
    """Get (or compute and cache) prediction for a golden tile."""
    if tile_name not in _prediction_cache:
        from predict_tile import predict_tile_from_arrays

        print(f"\n  Computing prediction for {tile_name}...")
        arrays = load_golden_tile(tile_name)
        pred = predict_tile_from_arrays(
            **arrays, model_path=str(MODEL_DIR), seed=42,
        )
        _prediction_cache[tile_name] = pred
        print(
            f"  {tile_name}: shape={pred.shape}, "
            f"mean={pred[pred != 255].mean():.1f}"
        )
    return _prediction_cache[tile_name]


def _get_stats(tile_name: str) -> dict:
    """Get (or compute and cache) parity stats for a golden tile."""
    if tile_name not in _stats_cache:
        pred = _get_prediction(tile_name)
        ref = load_reference_tif(tile_name)
        # Crop to overlapping region
        h = min(pred.shape[0], ref.shape[0])
        w = min(pred.shape[1], ref.shape[1])
        _stats_cache[tile_name] = compare_predictions(pred[:h, :w], ref[:h, :w])
    return _stats_cache[tile_name]


# ---------------------------------------------------------------------------
# Tier 1: Baseline — must not regress
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tile_name", GOLDEN_TILES)
def test_golden_baseline(tile_name):
    """Tier 1: baseline gates that must not regress."""
    stats = _get_stats(tile_name)

    print(f"\n  {tile_name} baseline:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    assert stats["n_valid"] > 0, "No valid pixels"
    assert stats["pct_within_10"] > 75, (
        f"Baseline: >75% within 10 DN, got {stats['pct_within_10']:.1f}%"
    )
    assert stats["pct_within_1"] > 40, (
        f"Baseline: >40% within 1 DN, got {stats['pct_within_1']:.1f}%"
    )


# ---------------------------------------------------------------------------
# Tier 2: Improved — target after cloud removal fixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tile_name", GOLDEN_TILES)
@pytest.mark.xfail(reason="Cloud removal not yet fully ported", strict=False)
def test_golden_improved(tile_name):
    """Tier 2: target after cloud removal improvements."""
    stats = _get_stats(tile_name)

    assert stats["pct_within_10"] > 95, (
        f"Improved: >95% within 10 DN, got {stats['pct_within_10']:.1f}%"
    )
    assert stats["pct_within_5"] > 90, (
        f"Improved: >90% within 5 DN, got {stats['pct_within_5']:.1f}%"
    )
    assert stats["pct_within_1"] > 70, (
        f"Improved: >70% within 1 DN, got {stats['pct_within_1']:.1f}%"
    )
    corr = stats.get("correlation_excl_outliers", stats["correlation"])
    assert corr > 0.75, f"Improved: corr >0.75, got {corr:.4f}"


# ---------------------------------------------------------------------------
# Tier 3: Target — production quality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tile_name", GOLDEN_TILES)
@pytest.mark.xfail(reason="Production parity not yet achieved", strict=False)
def test_golden_target(tile_name):
    """Tier 3: production quality target."""
    stats = _get_stats(tile_name)

    assert stats["pct_within_5"] > 99, (
        f"Target: >99% within 5 DN, got {stats['pct_within_5']:.1f}%"
    )
    assert stats["pct_within_1"] > 95, (
        f"Target: >95% within 1 DN, got {stats['pct_within_1']:.1f}%"
    )
    assert stats["correlation"] > 0.95, (
        f"Target: corr >0.95, got {stats['correlation']:.4f}"
    )
    assert stats["mean_abs_diff"] < 1.0, (
        f"Target: mean diff <1.0, got {stats['mean_abs_diff']:.2f}"
    )


# ---------------------------------------------------------------------------
# Aggregate report (runs after all tile tests)
# ---------------------------------------------------------------------------


def test_golden_aggregate_report():
    """Print aggregate parity report across all golden tiles."""
    # Ensure all tiles have been computed
    for tile in GOLDEN_TILES:
        _get_stats(tile)

    print_parity_table(_stats_cache)

    agg = aggregate_golden_report(_stats_cache)
    assert agg["mean_pct_within_10"] > 75, (
        f"Aggregate: mean >75% within 10 DN, got {agg['mean_pct_within_10']:.1f}%"
    )
