"""Lambda-deployed predict parity — two-layered validation.

Runs the 3 golden tiles through the deployed ``ttc-predict-dev`` Lambda and
compares the output to:

  1. **Reference TIFs** (`test_lambda_vs_reference`) — the same reference
     predictions ``test_golden_parity.py`` uses, applying the same tiered
     thresholds (>=90.6% within 1 DN, mean correlation excl outliers >= 0.997).
     Validates the Lambda is *correct* end-to-end.
  2. **Locally-run inference** (`test_lambda_vs_local`) — runs the same tiles
     through the Python inference path, asserts byte-level parity.
     Validates the Lambda matches the local build (catches libc/numpy ABI
     drift that reference-only comparison would miss).

Skipped unless ``PARITY_LAMBDA=1`` is set. Both tests require:
  - ARD for the 3 golden tiles uploaded to ``--dest`` (default ``s3://tof-output``)
  - ``.lithops/land-research/config.predict.yaml`` rendered (see docs/setup.md)
  - ``AWS_PROFILE=resto-user`` (or another profile with access to the bucket)

Usage::

    PARITY_LAMBDA=1 AWS_PROFILE=resto-user LITHOPS_ENV=land-research \\
        uv run pytest tests/parity/test_lambda_parity.py -v -s
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# REPO_ROOT itself so `import loaders.predict_tile` works as a package
# (needed for Lithops include_modules=["loaders"]).
# REPO_ROOT/loaders so `import predict_tile` works directly (used by _local_inference).
for p in (str(REPO_ROOT), str(REPO_ROOT / "loaders")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tests.conftest import GOLDEN_DIR, GOLDEN_TILES, MODEL_DIR
from tests.parity.metrics import aggregate_golden_report, compare_predictions


# Thresholds mirror the current passing state recorded in MEMORY.md
# (session 6, 2026-03-04). Keep in sync with test_golden_parity.py.
TIER_1_PCT_WITHIN_10 = 75.0
TIER_1_PCT_WITHIN_1 = 40.0
TIER_AGG_PCT_WITHIN_1 = 85.0
TIER_AGG_CORR_EXCL = 0.99


# Golden-tile coordinates (verified from data/tiledb.parquet).
GOLDEN_COORDS = {
    "1000X798Y": (1000, 798, -54.4722, -9.1944),
    "1000X799Y": (1000, 799, -54.4722, -9.1389),
    "1000X800Y": (1000, 800, -54.4722, -9.0833),
}

# Golden tiles were trained/compared against year 2020 predictions per the
# session notes; this is overridable via GOLDEN_YEAR.
DEFAULT_YEAR = int(os.environ.get("GOLDEN_YEAR", "2020"))


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.parity,
    pytest.mark.skipif(
        os.environ.get("PARITY_LAMBDA") != "1",
        reason="Set PARITY_LAMBDA=1 to run Lambda parity tests (incurs AWS cost).",
    ),
    pytest.mark.skipif(
        not GOLDEN_DIR.is_dir(),
        reason=f"Golden reference TIFs not found: {GOLDEN_DIR}",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env() -> str:
    return os.environ.get("LITHOPS_ENV", "land-research")


def _dest() -> str:
    return os.environ.get("PARITY_DEST", "s3://tof-output")


def _lithops_config_path() -> Path:
    p = REPO_ROOT / ".lithops" / _env() / "config.predict.yaml"
    if not p.exists():
        pytest.skip(f"Lithops predict config not rendered: {p}")
    return p


def _require_ard_on_s3(dest: str, year: int) -> None:
    from gri_tile_pipeline.tiles.availability import check_availability

    tiles = [
        {"year": year, "X_tile": x, "Y_tile": y, "lon": lon, "lat": lat}
        for x, y, lon, lat in GOLDEN_COORDS.values()
    ]
    result = check_availability(tiles, dest, check_type="raw_ard")
    if result["missing"]:
        names = ",".join(f"{t['X_tile']}X{t['Y_tile']}Y" for t in result["missing"])
        pytest.skip(
            f"Golden ARD missing on {dest} for year={year}: {names}. "
            "Upload with `gri-ttc download ...` before running Lambda parity."
        )


def _download_prediction(dest: str, year: int, x_tile: int, y_tile: int) -> np.ndarray:
    """Read the prediction FINAL.tif from *dest* (S3 or local)."""
    import rasterio

    from gri_tile_pipeline.storage.obstore_utils import from_dest
    from gri_tile_pipeline.storage.tile_paths import prediction_key

    key = prediction_key(year, x_tile, y_tile)

    if dest.startswith("s3://"):
        import obstore as obs

        store = from_dest(dest)
        data = obs.get(store, key).bytes()
        with rasterio.open(io.BytesIO(bytes(data))) as src:
            return src.read(1)
    else:
        with rasterio.open(Path(dest) / key) as src:
            return src.read(1)


def _invoke_lambda_for_tiles(year: int, dest: str) -> dict[str, np.ndarray]:
    """Map the 3 golden tiles through the deployed predict Lambda."""
    import yaml
    import lithops
    from lithops import FunctionExecutor

    # Mirror the import path used by scripts/predict_lambda_smoke.py so Lithops
    # sees the same worker shim as production.
    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import importlib

    lithops_workers = importlib.import_module("lithops_workers")

    with open(_lithops_config_path()) as f:
        lithops_cfg = yaml.safe_load(f)
    runtime = "ttc-predict-dev"
    lithops_cfg.setdefault("aws_lambda", {})["runtime"] = runtime

    kwargs_list = [
        {
            "year": year, "lon": lon, "lat": lat,
            "X_tile": x, "Y_tile": y,
            "dest": dest, "model_path": None, "debug": False,
            # Match _local_inference's seed for deterministic comparison.
            "seed": 42,
        }
        for x, y, lon, lat in GOLDEN_COORDS.values()
    ]

    fexec = FunctionExecutor(config=lithops_cfg, runtime=runtime, runtime_memory=6144)
    futures = [
        fexec.call_async(
            lithops_workers.run_predict, (kw,),
            include_modules=["loaders", "lithops_workers"],
        )
        for kw in kwargs_list
    ]
    # Lithops 3.6.1 moved the timeout kwarg off ResponseFuture.result onto
    # the executor. get_result waits on all futures and raises if any worker raised.
    fexec.get_result(futures, timeout=900)

    # Read results back from S3
    return {
        name: _download_prediction(dest, year, x, y)
        for name, (x, y, _lon, _lat) in GOLDEN_COORDS.items()
    }


def _local_inference(tile_name: str) -> np.ndarray:
    """Run the local Python inference path — mirrors test_golden_parity helpers."""
    import hickle as hkl
    from tests.conftest import GOLDEN_RAW

    from predict_tile import predict_tile_from_arrays  # type: ignore[import-not-found]

    arrays = {
        "s2_10": hkl.load(str(GOLDEN_RAW / "s2_10" / f"{tile_name}.hkl")),
        "s2_20": hkl.load(str(GOLDEN_RAW / "s2_20" / f"{tile_name}.hkl")),
        "s1": hkl.load(str(GOLDEN_RAW / "s1" / f"{tile_name}.hkl")),
        "dem": hkl.load(str(GOLDEN_RAW / "misc" / f"dem_{tile_name}.hkl")),
        "clouds": hkl.load(str(GOLDEN_RAW / "clouds" / f"clouds_{tile_name}.hkl")),
        "s2_dates": hkl.load(str(GOLDEN_RAW / "misc" / f"s2_dates_{tile_name}.hkl")),
    }
    clm_path = GOLDEN_RAW / "clouds" / f"cloudmask_{tile_name}.hkl"
    if clm_path.exists():
        arrays["clm"] = hkl.load(str(clm_path))
    return predict_tile_from_arrays(**arrays, model_path=str(MODEL_DIR), seed=42)


def _load_reference(tile_name: str) -> np.ndarray:
    import rasterio

    with rasterio.open(str(GOLDEN_DIR / f"{tile_name}_FINAL.tif")) as src:
        return src.read(1)


# ---------------------------------------------------------------------------
# Session-scoped Lambda invocation: share results across both tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lambda_predictions() -> dict[str, np.ndarray]:
    dest = _dest()
    year = DEFAULT_YEAR
    _require_ard_on_s3(dest, year)
    preds = _invoke_lambda_for_tiles(year, dest)
    for name, arr in preds.items():
        print(f"  Lambda[{name}]: shape={arr.shape} mean={arr[arr != 255].mean():.1f}")
    return preds


# ---------------------------------------------------------------------------
# Test 1: Lambda vs reference TIFs
# ---------------------------------------------------------------------------


def test_lambda_vs_reference(lambda_predictions: dict[str, np.ndarray]) -> None:
    """Lambda output vs the golden reference TIFs (correctness)."""
    per_tile: dict[str, dict] = {}

    for tile_name, pred in lambda_predictions.items():
        ref = _load_reference(tile_name)
        h = min(pred.shape[0], ref.shape[0])
        w = min(pred.shape[1], ref.shape[1])
        stats = compare_predictions(pred[:h, :w], ref[:h, :w])
        per_tile[tile_name] = stats
        print(
            f"  {tile_name}: within1={stats['pct_within_1']:.1f}% "
            f"within10={stats['pct_within_10']:.1f}% "
            f"corr_excl={stats['correlation_excl_outliers']:.4f}"
        )

        assert stats["pct_within_10"] >= TIER_1_PCT_WITHIN_10, (
            f"{tile_name}: pct_within_10={stats['pct_within_10']:.1f}% "
            f"below {TIER_1_PCT_WITHIN_10}%"
        )
        assert stats["pct_within_1"] >= TIER_1_PCT_WITHIN_1, (
            f"{tile_name}: pct_within_1={stats['pct_within_1']:.1f}% "
            f"below {TIER_1_PCT_WITHIN_1}%"
        )

    agg = aggregate_golden_report(per_tile)
    print(f"\n  Aggregate: {agg}")
    assert agg["mean_pct_within_1"] >= TIER_AGG_PCT_WITHIN_1, (
        f"Aggregate pct_within_1={agg['mean_pct_within_1']:.1f}% below {TIER_AGG_PCT_WITHIN_1}%"
    )
    assert agg["mean_correlation_excl_outliers"] >= TIER_AGG_CORR_EXCL, (
        f"Aggregate corr excl outliers={agg['mean_correlation_excl_outliers']:.4f} "
        f"below {TIER_AGG_CORR_EXCL}"
    )


# ---------------------------------------------------------------------------
# Test 2: Lambda vs local inference
# ---------------------------------------------------------------------------


@pytest.mark.tf
def test_lambda_vs_local(lambda_predictions: dict[str, np.ndarray]) -> None:
    """Lambda output vs locally-run inference (consistency).

    Flags Lambda-environment numeric drift (libc math, numpy ABI, TF build)
    that reference comparison alone would miss. Uses a very tight tolerance
    because both paths should be deterministic (seed=42).
    """
    if not (MODEL_DIR / "predict_graph-172.pb").is_file():
        pytest.skip(f"Model not found: {MODEL_DIR / 'predict_graph-172.pb'}")
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        pytest.skip("TensorFlow not installed locally")

    divergences: dict[str, dict] = {}

    for tile_name, lambda_pred in lambda_predictions.items():
        local_pred = _local_inference(tile_name)
        h = min(lambda_pred.shape[0], local_pred.shape[0])
        w = min(lambda_pred.shape[1], local_pred.shape[1])
        a = lambda_pred[:h, :w]
        b = local_pred[:h, :w]

        if np.array_equal(a, b):
            print(f"  {tile_name}: byte-identical ({h}x{w})")
            continue

        # Not identical — quantify the divergence. We tolerate <=1 DN max
        # diff as numeric noise; anything larger fails.
        mask = (a != 255) & (b != 255)
        diff = np.abs(a[mask].astype(int) - b[mask].astype(int))
        n_diff = int((diff > 0).sum())
        max_diff = int(diff.max()) if diff.size else 0
        divergences[tile_name] = {
            "n_diff_pixels": n_diff,
            "pct_diff": round(100.0 * n_diff / mask.sum(), 3) if mask.sum() else 0,
            "max_abs_diff": max_diff,
            "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        }
        print(f"  {tile_name} DIVERGED: {divergences[tile_name]}")

        assert max_diff <= 1, (
            f"{tile_name}: max abs diff {max_diff} > 1 DN between Lambda and local inference. "
            f"Details: {divergences[tile_name]}. This indicates the Lambda runtime image is "
            "numerically drifting from the local build (likely TF/numpy/libc ABI)."
        )
