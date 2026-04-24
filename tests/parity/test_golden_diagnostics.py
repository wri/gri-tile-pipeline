"""Diagnostic test for investigating golden tile parity gaps.

This is a manual-run diagnostic, not part of CI. It produces detailed
per-channel analysis and saves intermediate data to temp/diagnostics/.

Usage:
    pytest tests/parity/test_golden_diagnostics.py -v -s
    pytest tests/parity/test_golden_diagnostics.py -v -s -k "1000X798Y"
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import GOLDEN_DIR, GOLDEN_RAW, GOLDEN_TILES, MODEL_DIR
from tests.parity.metrics import compare_predictions, print_parity_table

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DIAG_DIR = REPO_ROOT / "temp" / "diagnostics"

_loader_path = str(REPO_ROOT / "loaders")
if _loader_path not in sys.path:
    sys.path.insert(0, _loader_path)

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_golden_exists = GOLDEN_DIR.is_dir()
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

CHANNEL_NAMES = [
    "B02", "B03", "B04", "B08",  # S2 10m
    "B05", "B06", "B07", "B8A", "B11", "B12",  # S2 20m
    "DEM",
    "S1_VV", "S1_VH",
    "EVI", "BI", "MSAVI2", "GRNDVI",
]


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
    import rasterio

    with rasterio.open(str(GOLDEN_DIR / f"{tile_name}_FINAL.tif")) as src:
        return src.read(1)


@pytest.mark.parametrize("tile_name", GOLDEN_TILES)
def test_golden_diagnostics(tile_name):
    """Run diagnostic analysis on a golden tile: cloud removal on vs off."""
    from loaders.predict_tile import predict_tile_from_arrays

    tile_diag_dir = DIAG_DIR / tile_name
    os.makedirs(tile_diag_dir, exist_ok=True)

    ref = load_reference_tif(tile_name)
    arrays = load_golden_tile(tile_name)

    results = {}

    for cloud_label, enable_cloud in [("cloud_on", True), ("cloud_off", False)]:
        print(f"\n  --- {tile_name} / {cloud_label} ---")
        diag = {}
        intermediates = {}
        pred = predict_tile_from_arrays(
            **arrays,
            model_path=str(MODEL_DIR),
            enable_cloud_removal=enable_cloud,
            diagnostics=diag,
            intermediates=intermediates,
            seed=42,
        )

        # Crop to overlapping region
        h = min(pred.shape[0], ref.shape[0])
        w = min(pred.shape[1], ref.shape[1])
        stats = compare_predictions(pred[:h, :w], ref[:h, :w])

        results[cloud_label] = {"stats": stats, "diagnostics": diag, "intermediates": intermediates}

        # Save intermediates
        inter_dir = tile_diag_dir / f"intermediates_{cloud_label}"
        os.makedirs(inter_dir, exist_ok=True)
        for key, arr in intermediates.items():
            if isinstance(arr, np.ndarray):
                np.save(str(inter_dir / f"{key}.npy"), arr)
                print(f"    intermediate {key}: shape={arr.shape}, mean={arr.mean():.6f}")

        # Print parity stats
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

        # Print feature stack channel stats
        if "feature_stack_channel_stats" in diag:
            print(f"\n  Feature stack channel stats ({cloud_label}):")
            for ch_idx, ch_stat in diag["feature_stack_channel_stats"].items():
                ch_name = CHANNEL_NAMES[int(ch_idx)] if int(ch_idx) < len(CHANNEL_NAMES) else f"ch{ch_idx}"
                print(
                    f"    {ch_name:>8}: min={ch_stat['min']:.4f}  "
                    f"max={ch_stat['max']:.4f}  mean={ch_stat['mean']:.4f}"
                )

        # Save difference map
        diff_map = np.abs(pred[:h, :w].astype(float) - ref[:h, :w].astype(float))
        mask = (pred[:h, :w] == 255) | (ref[:h, :w] == 255)
        diff_map[mask] = np.nan
        np.save(str(tile_diag_dir / f"diff_map_{cloud_label}.npy"), diff_map)

    # Compare cloud on vs off
    on_pct1 = results["cloud_on"]["stats"].get("pct_within_1", 0)
    off_pct1 = results["cloud_off"]["stats"].get("pct_within_1", 0)
    print(f"\n  Cloud removal impact on %<=1DN: on={on_pct1:.1f}%, off={off_pct1:.1f}%")
    print(f"  Delta: {on_pct1 - off_pct1:+.1f}% (positive = cloud_on is better)")

    # Save diagnostics JSON
    serializable = {}
    for label, data in results.items():
        serializable[label] = {
            "stats": data["stats"],
            "diagnostics": {
                k: v if not isinstance(v, tuple) else list(v)
                for k, v in data["diagnostics"].items()
            },
        }
    with open(tile_diag_dir / "diagnostics.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\n  Saved diagnostics to {tile_diag_dir}/")
