"""End-to-end prediction parity test against reference TIF.

Uses tiered quality gates that track progress as cloud removal is ported.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import ARD_DIR, MODEL_DIR, REFERENCE_TIF, has_ard, has_model, has_reference, has_tf
from tests.parity.metrics import compare_predictions

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

pytestmark = [pytest.mark.parity, pytest.mark.slow, pytest.mark.tf]


@has_ard
@has_reference
@has_model
@has_tf
def test_prediction_parity_baseline():
    """Baseline tier: current state without full cloud removal."""
    import importlib
    import rasterio

    loader_path = str(REPO_ROOT / "loaders")
    if loader_path not in sys.path:
        sys.path.insert(0, loader_path)
    # Force reimport in case of caching
    import predict_tile
    importlib.reload(predict_tile)
    run_local = predict_tile.run_local

    with rasterio.open(str(REFERENCE_TIF)) as src:
        ref = src.read(1)

    output_path = os.path.join(str(REPO_ROOT), "temp", "test_parity_output.tif")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ours = run_local(
        ard_dir=str(ARD_DIR),
        model_path=str(MODEL_DIR),
        output_path=output_path,
    )

    h = min(ref.shape[0], ours.shape[0])
    w = min(ref.shape[1], ours.shape[1])
    stats = compare_predictions(ours[:h, :w], ref[:h, :w])

    print(f"\n--- Parity Results ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Baseline tier gates (current state with missing cloud removal)
    assert stats["n_valid"] > 0, "No valid pixels for comparison"
    assert stats["pct_within_1"] > 50, (
        f"Baseline: >50% within 1 DN required, got {stats['pct_within_1']:.1f}%"
    )
    assert stats["pct_within_10"] > 90, (
        f"Baseline: >90% within 10 DN required, got {stats['pct_within_10']:.1f}%"
    )

    # Correlation excluding outliers (>30 DN diff are cloud-affected)
    mask = (ref[:h, :w] != 255) & (ours[:h, :w] != 255)
    diff = np.abs(ref[:h, :w][mask].astype(float) - ours[:h, :w][mask].astype(float))
    non_outlier = diff <= 30
    if non_outlier.sum() > 100:
        corr_excl = np.corrcoef(
            ref[:h, :w][mask][non_outlier].astype(float),
            ours[:h, :w][mask][non_outlier].astype(float),
        )[0, 1]
        print(f"  correlation_excl_outliers: {corr_excl:.4f}")
        assert corr_excl > 0.70, (
            f"Baseline: correlation >0.70 (excl outliers) required, got {corr_excl:.4f}"
        )
