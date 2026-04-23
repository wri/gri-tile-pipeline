"""Single-tile round-trip viability test for the deployed predict Lambda.

Runs a single prediction through the ``ttc-predict-dev`` Lambda via Lithops,
then validates the resulting ``FINAL.tif`` has a sensible shape, dtype, and
content distribution. Intended as the first thing to run after
``make -C infra build-all ENV=land-research`` to confirm the runtime image,
IAM role, and cross-account ``tof-output`` write are all wired correctly.

Usage (from repo root, after ``aws sso login --profile resto-user``)::

    AWS_PROFILE=resto-user LITHOPS_ENV=land-research \\
        uv run python scripts/predict_lambda_smoke.py

Exit codes:
    0 — Lambda invoked, valid TIF produced, assertions pass
    2 — ARD missing on S3 for the selected tile (precondition failure)
    3 — Lambda invocation failed
    4 — Output TIF missing or failed validity assertions
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml
from loguru import logger

# Known-good golden tiles with ARD already uploaded.
KNOWN_TILES: dict[str, dict] = {
    "1000X871Y": {"X_tile": 1000, "Y_tile": 871, "lon": -54.4722, "lat": -5.1389},
    "1000X798Y": {"X_tile": 1000, "Y_tile": 798, "lon": -54.4722, "lat": -9.1944},
    "1000X799Y": {"X_tile": 1000, "Y_tile": 799, "lon": -54.4722, "lat": -9.1389},
    "1000X800Y": {"X_tile": 1000, "Y_tile": 800, "lon": -54.4722, "lat": -9.0833},
}


def _resolve_tile(tile_arg: str, lookup_parquet: str | None) -> dict:
    """Resolve a tile argument to (X_tile, Y_tile, lon, lat).

    ``tile_arg`` is either a label like ``1000X871Y`` that we have hard-coded
    or ``{X}X{Y}Y`` + a ``--lookup-parquet`` to find it in tiledb.
    """
    if tile_arg in KNOWN_TILES:
        return KNOWN_TILES[tile_arg]
    import re

    m = re.match(r"^(\d+)X(\d+)Y$", tile_arg)
    if not m:
        raise ValueError(
            f"Tile '{tile_arg}' not in the known list and is not shaped like '{{X}}X{{Y}}Y'."
        )
    x_tile, y_tile = int(m.group(1)), int(m.group(2))
    if not lookup_parquet or not Path(lookup_parquet).exists():
        raise ValueError(
            f"Tile {tile_arg} is not in the known list; pass --lookup-parquet data/tiledb.parquet "
            "so the smoke test can look up its lon/lat."
        )
    import duckdb

    con = duckdb.connect()
    row = con.execute(
        "SELECT X, Y FROM read_parquet(?) WHERE X_tile = ? AND Y_tile = ?",
        [lookup_parquet, x_tile, y_tile],
    ).fetchone()
    if not row:
        raise ValueError(f"Tile {tile_arg} not found in {lookup_parquet}")
    return {"X_tile": x_tile, "Y_tile": y_tile, "lon": row[0], "lat": row[1]}


def _check_ard(tile: dict, year: int, dest: str) -> None:
    from gri_tile_pipeline.tiles.availability import check_availability

    probe = dict(tile)
    probe["year"] = year
    result = check_availability([probe], dest, check_type="raw_ard")
    if result["missing"]:
        raise SystemExit(
            f"[PRECONDITION] ARD missing for tile {tile['X_tile']}X{tile['Y_tile']}Y "
            f"year={year} at {dest}. Upload ARD first (gri-ttc download ...), or pick "
            "a different --tile. Exiting 2."
        )


def _invoke_predict(
    kwargs: dict, predict_cfg_path: str, runtime: str, memory_mb: int, timeout: int,
) -> float:
    """Submit one Lambda invocation and block until the future resolves.

    Returns wall-clock seconds (cold-start + inference + write) as measured
    from submission to result retrieval.
    """
    import lithops
    from lithops import FunctionExecutor

    with open(predict_cfg_path) as f:
        lithops_cfg = yaml.safe_load(f)
    lithops_cfg.setdefault("aws_lambda", {})["runtime"] = runtime

    # Import the same thin worker shim the production code uses so we test
    # the real call path (not a direct import of loaders.predict_tile.run).
    import importlib
    import sys as _sys

    # Put the repo root and src/ on sys.path so Lithops' include_modules can
    # locate and bundle `loaders` (at repo root) and `lithops_workers` (in src/).
    # Without this, `uv run python scripts/...` puts only scripts/ on sys.path.
    repo_root = str(Path(__file__).resolve().parents[1])
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    for p in (repo_root, src_dir):
        if p not in _sys.path:
            _sys.path.insert(0, p)
    lithops_workers = importlib.import_module("lithops_workers")

    fexec = FunctionExecutor(
        config=lithops_cfg, runtime=runtime, runtime_memory=memory_mb,
    )
    t0 = time.time()
    future = fexec.call_async(
        lithops_workers.run_predict, (kwargs,),
        include_modules=["loaders", "lithops_workers"],
    )
    # get_result blocks and raises if the worker raised. Lithops 3.6.1
    # moved the timeout kwarg off ResponseFuture.result onto the executor,
    # and get_result expects a list-like fs (it calls len() on it).
    fexec.get_result([future], timeout=timeout)
    return time.time() - t0


def _validate_tif(dest: str, year: int, tile: dict) -> dict:
    import io

    import numpy as np
    import rasterio

    from gri_tile_pipeline.storage.obstore_utils import from_dest
    from gri_tile_pipeline.storage.tile_paths import prediction_key

    key = prediction_key(year, tile["X_tile"], tile["Y_tile"])

    if dest.startswith("s3://"):
        import obstore as obs

        store = from_dest(dest)
        data = obs.get(store, key).bytes()
        buf = io.BytesIO(bytes(data))
        src = rasterio.open(buf)
    else:
        src = rasterio.open(Path(dest) / key)

    with src:
        arr = src.read(1)
        profile = src.profile

    if arr.dtype != np.uint8:
        raise SystemExit(f"[FAIL] Output dtype {arr.dtype} (expected uint8). Exiting 4.")
    h, w = arr.shape
    # Reference tiles are ~618x616 (4x4 subtile grid, SIZE=158 minus borders).
    # Anything dramatically smaller indicates truncated output.
    if h < 500 or w < 500:
        raise SystemExit(f"[FAIL] Output too small: {h}x{w} (expected ~618x616). Exiting 4.")

    n_total = arr.size
    n_nodata = int((arr == 255).sum())
    n_valid = int((arr < 255).sum())
    n_in_range = int(((arr >= 1) & (arr <= 99)).sum())

    if n_nodata == n_total:
        raise SystemExit("[FAIL] All pixels are nodata (255). Model ran but produced no valid output. Exiting 4.")
    if n_in_range == 0:
        raise SystemExit(
            "[FAIL] No pixels in [1,99] — output is only saturated 0/100/255. Exiting 4."
        )

    return {
        "shape": [int(h), int(w)],
        "n_total": int(n_total),
        "n_nodata": n_nodata,
        "n_valid": n_valid,
        "n_in_range_1_99": n_in_range,
        "pct_nodata": round(100.0 * n_nodata / n_total, 2),
        "mean_tree_cover": float(round(float(arr[arr < 255].mean()), 2)) if n_valid else None,
        "crs": str(profile.get("crs")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--env", default=os.environ.get("LITHOPS_ENV", "land-research"),
        help="Lithops env dir under .lithops/ (default: $LITHOPS_ENV or 'land-research').",
    )
    ap.add_argument("--tile", default="1000X871Y", help="Tile label (e.g. 1000X871Y).")
    ap.add_argument("--year", type=int, default=2023, help="Prediction year.")
    ap.add_argument("--dest", default="s3://tof-output", help="S3 URI or local path for ARD + output TIF.")
    ap.add_argument("--memory-mb", type=int, default=6144, help="Lambda memory.")
    ap.add_argument("--timeout", type=int, default=900, help="Client-side wait timeout (sec); should be >= aws_lambda.runtime_timeout.")
    ap.add_argument("--runtime", default="ttc-predict-dev", help="Lithops runtime name.")
    ap.add_argument("--lookup-parquet", default="data/tiledb.parquet",
                    help="Parquet tile lookup (used only if --tile is not in the known list).")
    args = ap.parse_args()

    logger.info(f"Smoke test: env={args.env} tile={args.tile} year={args.year} dest={args.dest}")

    tile = _resolve_tile(args.tile, args.lookup_parquet)
    logger.info(
        f"Resolved tile: X_tile={tile['X_tile']} Y_tile={tile['Y_tile']} "
        f"lon={tile['lon']} lat={tile['lat']}"
    )

    _check_ard(tile, args.year, args.dest)
    logger.info("ARD precondition OK")

    predict_cfg = Path(".lithops") / args.env / "config.predict.yaml"
    if not predict_cfg.exists():
        logger.error(
            f"Lithops config not found: {predict_cfg}. Run 'make -C infra render ENV={args.env}'."
        )
        return 3

    kwargs = {
        "year": args.year,
        "lon": tile["lon"],
        "lat": tile["lat"],
        "X_tile": tile["X_tile"],
        "Y_tile": tile["Y_tile"],
        "dest": args.dest,
        "model_path": None,
        "debug": False,
        "seed": 42,
    }

    try:
        elapsed = _invoke_predict(
            kwargs, str(predict_cfg), args.runtime, args.memory_mb, args.timeout,
        )
    except Exception as e:
        logger.error(f"Lambda invocation failed: {e}")
        return 3

    logger.info(f"Lambda completed in {elapsed:.1f}s (includes any cold start)")

    stats = _validate_tif(args.dest, args.year, tile)
    logger.info(f"Output TIF stats: {stats}")
    logger.info("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
