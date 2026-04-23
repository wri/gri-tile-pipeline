"""Parity report for any S3-hosted tile against its existing prediction.

Given a tile that already has complete ARD **and** a ``FINAL.tif`` prediction on
S3, this script re-runs the prediction pipeline and compares the fresh output
to the existing TIF — without overwriting anything on S3. Two modes:

  * ``--mode local`` (default) — runs ``predict_tile_from_arrays`` in-process.
    Fast, no AWS cost, isolates logic drift.

  * ``--mode lambda`` — invokes the deployed ``ttc-predict-dev`` Lambda via
    Lithops with ``prediction_key_override`` so the output lands in a scratch
    key that's deleted after the comparison. Reports wall-clock, Lambda
    billed duration, cold-start flag, and an estimated cost in USD from the
    function's memory × billed duration.

Both modes produce the same self-contained HTML report with blink comparator,
side-by-side / diff / outlier maps, histograms, and the full metrics table
(reusing ``tests/parity/generate_report.py`` visuals).

Usage (from repo root)::

    AWS_PROFILE=dl-user uv run python scripts/tile_parity_report.py \\
        --tile 1274X655Y --year 2025

    AWS_PROFILE=dl-user LITHOPS_ENV=datalab-test uv run python \\
        scripts/tile_parity_report.py --tile 1274X655Y --year 2025 --mode lambda

Exits 0 on success regardless of parity result — this is a diagnostic, not a
gate. The HTML report contains the numbers.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Repo root for `import loaders.predict_tile` (package form).
# loaders/ so `import predict_tile` works (used by tests/parity/generate_report.py).
# tests/ so the parity helpers import cleanly.
for p in (REPO_ROOT, REPO_ROOT / "loaders", REPO_ROOT / "tests"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import numpy as np
from loguru import logger


_TILE_RE = re.compile(r"^(\d+)X(\d+)Y$")

# ARD keys in the predictions bucket have drifted over time across two orthogonal axes:
#   1. bucket prefix: <root> vs 'dev-ttc-lithops-usw2/' (and maybe others)
#   2. base pattern: canonical '{year}/raw/{x}/{y}/raw/...' vs alternate
#      '{year}/{x}/{y}/raw/...' (the first 'raw/' segment is missing).
# For a single tile, different sources may use different combinations —
# e.g. s1 under alt pattern + usw2 prefix, dem under canonical + usw2 prefix.
# So we probe per-source and pick the first candidate that exists.
#
# Ordering matters: earlier candidates win. Canonical/root first, then add
# the known historical combos.
ARD_KEY_VARIANTS: tuple[tuple[str, str], ...] = (
    # (prefix, base_pattern)
    ("",                          "{year}/raw/{x}/{y}/raw"),
    ("",                          "{year}/{x}/{y}/raw"),
    ("dev-ttc-lithops-usw2/",     "{year}/raw/{x}/{y}/raw"),
    ("dev-ttc-lithops-usw2/",     "{year}/{x}/{y}/raw"),
)


# --- AWS Lambda pricing (us-east-1, x86_64, standard tier) ---
# Values are from AWS public pricing; update the constant + PRICING_STAMP
# when AWS publishes a new rate card. See:
#   https://aws.amazon.com/lambda/pricing/
LAMBDA_PRICE_PER_GB_SEC = 0.0000166667
LAMBDA_PRICE_PER_REQUEST = 0.0000002  # $0.20 per 1M requests
PRICING_STAMP = "x86 us-east-1 standard tier (2026-04)"


def _parse_tile(arg: str) -> tuple[int, int]:
    m = _TILE_RE.match(arg)
    if not m:
        raise argparse.ArgumentTypeError(f"Tile '{arg}' must be shaped like '1274X655Y'.")
    return int(m.group(1)), int(m.group(2))


def _ard_suffix(src: str, year: int, x: int, y: int) -> str:
    """Return the per-source subpath (e.g. 'misc/dem_1000X798Y.hkl') by
    stripping the canonical base from raw_ard_keys_by_source's output."""
    from gri_tile_pipeline.storage.tile_paths import raw_ard_keys_by_source

    canonical_base = f"{year}/raw/{x}/{y}/raw/"
    full = raw_ard_keys_by_source(year, x, y)[src]
    if not full.startswith(canonical_base):
        raise RuntimeError(f"Unexpected canonical key shape: {full!r}")
    return full[len(canonical_base):]


def _ard_key_candidates(
    year: int, x: int, y: int, src: str,
    variants: tuple[tuple[str, str], ...] = ARD_KEY_VARIANTS,
) -> list[str]:
    """Every (prefix × base-pattern) S3 key to probe for a single ARD source."""
    suffix = _ard_suffix(src, year, x, y)
    out: list[str] = []
    for prefix, base_tmpl in variants:
        base = base_tmpl.format(year=year, x=x, y=y)
        out.append(f"{prefix}{base}/{suffix}")
    return out


def _variant_label(key: str, year: int, x: int, y: int) -> str:
    """Identify which (prefix, base-pattern) matched a resolved key, for logging."""
    for prefix, base_tmpl in ARD_KEY_VARIANTS:
        base = base_tmpl.format(year=year, x=x, y=y)
        if key.startswith(f"{prefix}{base}/"):
            p = prefix or "<root>"
            return f"{p}|{base_tmpl}"
    return "<unknown-variant>"


def _resolve_ard_keys(
    store, year: int, x: int, y: int,
    variants: tuple[tuple[str, str], ...] = ARD_KEY_VARIANTS,
) -> dict[str, str]:
    """Resolve each ARD source independently to the first candidate that exists.

    Returns ``{source: full_key}``. Raises ``SystemExit`` if any source cannot
    be located under any variant — the error lists every probed key so it's
    clear whether a new prefix/pattern is needed or the ARD is genuinely
    missing.
    """
    import obstore as obs
    from gri_tile_pipeline.storage.tile_paths import raw_ard_keys_by_source

    sources = list(raw_ard_keys_by_source(year, x, y).keys())
    resolved: dict[str, str] = {}
    attempts: dict[str, list[str]] = {}

    for src in sources:
        attempts[src] = _ard_key_candidates(year, x, y, src, variants=variants)
        for key in attempts[src]:
            try:
                obs.head(store, key)
            except FileNotFoundError:
                continue
            resolved[src] = key
            break

    missing = [s for s in sources if s not in resolved]
    if missing:
        lines = [f"[PRECONDITION] Could not locate ARD for {x}X{y}Y year={year}:"]
        for s in missing:
            lines.append(f"  {s} probed (all missing):")
            for k in attempts[s]:
                lines.append(f"    - {k}")
        found = [s for s in sources if s in resolved]
        if found:
            lines.append(f"  (found: {', '.join(found)})")
        lines.append("Exiting 2.")
        raise SystemExit("\n".join(lines))

    # Log resolution summary so mixed variants are visible.
    from collections import Counter

    by_variant = Counter(_variant_label(k, year, x, y) for k in resolved.values())
    if len(by_variant) == 1:
        logger.info(f"ARD resolved, all 6 sources under {next(iter(by_variant))}")
    else:
        logger.info("ARD resolved from MIXED variants:")
        for src, key in resolved.items():
            logger.info(f"  {src:<10} {_variant_label(key, year, x, y)}")
    return resolved


def _verify_prediction_exists(store, year: int, x: int, y: int) -> str:
    """Ensure FINAL.tif exists at the bucket root and return its key."""
    import obstore as obs

    from gri_tile_pipeline.storage.tile_paths import prediction_key

    key = prediction_key(year, x, y)
    try:
        obs.head(store, key)
    except FileNotFoundError:
        raise SystemExit(
            f"[PRECONDITION] Existing prediction missing: {key}. "
            "This tool requires a tile with an existing FINAL.tif to compare against. Exiting 2."
        )
    return key


def _load_ard(store, resolved_keys: dict[str, str]) -> dict[str, np.ndarray]:
    """Download the six ARD files from their resolved S3 keys."""
    from loaders.predict_tile import _load_hkl

    logger.info(f"Loading ARD from S3 ({len(resolved_keys)} sources)")
    arrays = {src: _load_hkl(store, key) for src, key in resolved_keys.items()}
    for src, arr in arrays.items():
        logger.debug(
            f"  {src}: shape={getattr(arr, 'shape', '?')} dtype={getattr(arr, 'dtype', '?')}"
        )
    return arrays


def _load_existing_prediction(store, year: int, x: int, y: int) -> np.ndarray:
    """Read the existing ``FINAL.tif`` from S3 into a 2D uint8 array."""
    import io

    import obstore as obs
    import rasterio

    from gri_tile_pipeline.storage.tile_paths import prediction_key

    key = prediction_key(year, x, y)
    logger.info(f"Downloading existing prediction: {key}")
    data = bytes(obs.get(store, key).bytes())
    with rasterio.open(io.BytesIO(data)) as src:
        return src.read(1)


def _run_local_predict(ard: dict[str, np.ndarray], model_dir: Path, seed: int) -> dict:
    """Run predict_tile_from_arrays locally. Returns {pred, execution} where
    ``execution`` is a mode-tagged dict consumed by the report builder."""
    from loaders.predict_tile import predict_tile_from_arrays

    logger.info(f"Running local prediction with seed={seed} model_dir={model_dir}")
    t0 = time.time()
    pred = predict_tile_from_arrays(
        ard["s2_10"], ard["s2_20"], ard["s1"], ard["dem"],
        ard["clouds"], ard["s2_dates"],
        str(model_dir),
        seed=seed,
    )
    elapsed = time.time() - t0
    logger.info(f"Local prediction done: shape={pred.shape} in {elapsed:.1f}s")
    return {
        "pred": pred,
        "execution": {"mode": "local", "wall_clock_sec": elapsed, "seed": seed},
    }


def _compute_lambda_cost(memory_mb: int, billed_sec: float) -> dict:
    """AWS Lambda cost estimate. Memory in MiB, duration in seconds."""
    gb_sec = (memory_mb / 1024.0) * billed_sec
    compute_usd = gb_sec * LAMBDA_PRICE_PER_GB_SEC
    total_usd = compute_usd + LAMBDA_PRICE_PER_REQUEST
    return {
        "gb_sec": gb_sec,
        "compute_usd": compute_usd,
        "request_usd": LAMBDA_PRICE_PER_REQUEST,
        "total_usd": total_usd,
        "pricing_stamp": PRICING_STAMP,
    }


def _invoke_lambda(
    year: int, x: int, y: int,
    dest: str,
    resolved_keys: dict[str, str],
    scratch_key: str,
    seed: int,
    env: str,
    memory_mb: int,
    timeout_sec: int = 900,
) -> dict:
    """Invoke the deployed ttc-predict-dev Lambda and return execution metadata.

    Writes to ``scratch_key`` (never the production prediction key), uses the
    caller-resolved ARD map so non-canonical tiles work. Returns a dict that
    mirrors the ``_run_local_predict`` execution shape plus Lambda-specific
    fields (``worker_exec_time_sec``, ``cold_start``, ``cost``).
    """
    import importlib

    import yaml
    from lithops import FunctionExecutor

    # Lithops' include_modules imports these client-side to bundle, so we need
    # both src/ (for lithops_workers) and the repo root (for `loaders` as a package).
    for p in (REPO_ROOT, REPO_ROOT / "src"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    lithops_workers = importlib.import_module("lithops_workers")

    cfg_path = REPO_ROOT / ".lithops" / env / "config.predict.yaml"
    if not cfg_path.exists():
        raise SystemExit(
            f"[CONFIG] Lithops predict config not rendered: {cfg_path}. "
            f"Run `make -C infra render ENV={env}` first. Exiting 3."
        )
    with cfg_path.open() as f:
        lithops_cfg = yaml.safe_load(f)
    runtime = "ttc-predict-dev"
    lithops_cfg.setdefault("aws_lambda", {})["runtime"] = runtime

    # lon/lat only affect GeoTIFF metadata (not pixel values). For parity we
    # compare pixels, so 0/0 is fine for the scratch write.
    kwargs = {
        "year": year, "lon": 0.0, "lat": 0.0,
        "X_tile": x, "Y_tile": y,
        "dest": dest,
        "model_path": None, "debug": False, "seed": seed,
        "prediction_key_override": scratch_key,
        "ard_keys_override": resolved_keys,
    }

    fexec = FunctionExecutor(
        config=lithops_cfg, runtime=runtime, runtime_memory=memory_mb,
    )
    logger.info(
        f"Invoking Lambda ttc-predict-dev (memory={memory_mb}MB, timeout={timeout_sec}s), "
        f"scratch_key={scratch_key}"
    )
    t0 = time.time()
    fut = fexec.call_async(
        lithops_workers.run_predict, (kwargs,),
        include_modules=["loaders", "lithops_workers"],
    )
    fexec.get_result([fut], timeout=timeout_sec)
    wall = time.time() - t0

    # Lithops stats (best-effort — field names have changed between minor versions).
    stats: dict = dict(getattr(fut, "stats", {}) or {})
    worker_sec = (
        stats.get("worker_func_exec_time")
        or stats.get("worker_exec_time")
        or (stats.get("worker_end_tstamp", 0) - stats.get("worker_start_tstamp", 0))
        or 0.0
    )
    cold_start = bool(stats.get("worker_cold_start", False))

    # Lambda bills what actually ran; use worker_exec_time if present, else fall
    # back to wall clock (slight over-estimate since it includes Lithops overhead).
    billed_sec = float(worker_sec) if worker_sec else wall
    cost = _compute_lambda_cost(memory_mb, billed_sec)

    logger.info(
        f"Lambda done: wall={wall:.1f}s  worker_exec={worker_sec:.1f}s  "
        f"cold_start={cold_start}  memory={memory_mb}MB  "
        f"est_cost=${cost['total_usd']:.4f}"
    )
    return {
        "execution": {
            "mode": "lambda",
            "wall_clock_sec": wall,
            "worker_exec_time_sec": float(worker_sec),
            "billed_sec": billed_sec,
            "cold_start": cold_start,
            "memory_mb": memory_mb,
            "seed": seed,
            "lithops_stats": stats,
            "cost": cost,
            "scratch_key": scratch_key,
            "lithops_env": env,
        },
    }


def _download_scratch_prediction(store, scratch_key: str) -> np.ndarray:
    """Read the Lambda's scratch FINAL.tif back into a uint8 array."""
    import io

    import obstore as obs
    import rasterio

    logger.info(f"Downloading Lambda scratch output: {scratch_key}")
    data = bytes(obs.get(store, scratch_key).bytes())
    with rasterio.open(io.BytesIO(data)) as src:
        return src.read(1)


def _delete_scratch(store, scratch_key: str) -> None:
    import obstore as obs

    try:
        obs.delete(store, scratch_key)
        logger.info(f"Deleted scratch: {scratch_key}")
    except Exception as e:
        logger.warning(f"Failed to delete scratch {scratch_key}: {e}")


def _execution_card_html(execution: dict) -> str:
    """Render an execution/cost card tailored to local vs lambda mode."""
    mode = execution["mode"]
    wall = execution["wall_clock_sec"]
    if mode == "local":
        return f"""
    <div class="summary-section">
      <h3>Execution</h3>
      <table class="metrics-table">
        <tr><th>Mode</th><td>local (<code>predict_tile_from_arrays</code>, in-process)</td></tr>
        <tr><th>Wall-clock</th><td>{wall:.1f} s</td></tr>
        <tr><th>Seed</th><td>{execution['seed']}</td></tr>
      </table>
    </div>
    """

    cost = execution["cost"]
    worker = execution["worker_exec_time_sec"]
    billed = execution["billed_sec"]
    cold = "yes" if execution["cold_start"] else "no"
    overhead = max(wall - worker, 0.0) if worker else 0.0
    return f"""
    <div class="summary-section">
      <h3>Execution — Lambda</h3>
      <table class="metrics-table">
        <tr><th>Mode</th><td>lambda (<code>ttc-predict-dev</code> via Lithops)</td></tr>
        <tr><th>Lithops env</th><td><code>{execution['lithops_env']}</code></td></tr>
        <tr><th>Scratch key</th><td><code>{execution['scratch_key']}</code> (deleted after run)</td></tr>
        <tr><th>Memory</th><td>{execution['memory_mb']} MB</td></tr>
        <tr><th>Wall-clock (client)</th><td>{wall:.1f} s</td></tr>
        <tr><th>Worker exec time</th><td>{worker:.2f} s</td></tr>
        <tr><th>Lithops overhead</th><td>{overhead:.2f} s</td></tr>
        <tr><th>Cold start</th><td>{cold}</td></tr>
        <tr><th>Seed</th><td>{execution['seed']}</td></tr>
      </table>
      <h3 style="margin-top:14px">Cost estimate</h3>
      <table class="metrics-table">
        <tr><th>Billed duration</th><td>{billed:.2f} s
            <span style="color:#888">(worker_exec_time or wall fallback)</span></td></tr>
        <tr><th>GB-seconds</th><td>{cost['gb_sec']:.2f}</td></tr>
        <tr><th>Compute cost</th><td>${cost['compute_usd']:.6f}</td></tr>
        <tr><th>Request cost</th><td>${cost['request_usd']:.6f}</td></tr>
        <tr style="font-weight:600"><th>Total</th><td>${cost['total_usd']:.6f}</td></tr>
        <tr><th>Pricing</th><td style="color:#888">{cost['pricing_stamp']}</td></tr>
      </table>
      <p style="color:#888; font-size:12px; margin-top:6px">
        Scale: $0.001/tile × 1,000 tiles = $1. × 1,000,000 tiles ≈ ${cost['total_usd'] * 1_000_000:.0f}.
      </p>
    </div>
    """


def _build_report(
    tile_label: str,
    year: int,
    new_pred: np.ndarray,
    existing_pred: np.ndarray,
    output_path: Path,
    execution: dict,
) -> dict:
    """Write the HTML report; return the metrics dict."""
    # Reuse the battle-tested visuals from generate_report.py.
    from parity.generate_report import HTML_TEMPLATE, make_tile_section
    from parity.metrics import compare_predictions

    # Align dims defensively — downloads sometimes differ by a row/col at edges.
    h = min(new_pred.shape[0], existing_pred.shape[0])
    w = min(new_pred.shape[1], existing_pred.shape[1])
    new_c = new_pred[:h, :w]
    ex_c = existing_pred[:h, :w]

    stats = compare_predictions(new_c, ex_c)

    # make_tile_section casts "ours" = first arg (new prediction), "reference"
    # = second arg (existing S3).
    section = make_tile_section(f"{tile_label} (year={year})", new_c, ex_c, stats)

    mode = execution["mode"]
    mode_desc = (
        "local in-process run via <code>predict_tile_from_arrays</code>"
        if mode == "local"
        else "deployed Lambda <code>ttc-predict-dev</code> via Lithops (scratch key; deleted after)"
    )
    summary = f"""
    <div class="summary-section">
      <h2>Summary</h2>
      <p><b>Tile:</b> {tile_label} &nbsp; <b>Year:</b> {year} &nbsp;
         <b>Shape:</b> {h} x {w}px</p>
      <p><b>New prediction:</b> {mode_desc} (seed={execution['seed']}), no S3 overwrite.</p>
      <p><b>Existing prediction:</b> downloaded from S3 <code>FINAL.tif</code>,
         treated as the "reference" in the metrics table below.</p>
    </div>
    {_execution_card_html(execution)}
    """

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    html = HTML_TEMPLATE.format(timestamp=timestamp, summary=summary, tiles=section)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    logger.info(
        f"Report written: {output_path}  ({output_path.stat().st_size / 1024:.0f} KB)"
    )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--tile", required=True, type=str,
                    help="Tile label like '1274X655Y'.")
    ap.add_argument("--year", required=True, type=int,
                    help="Year (matches the S3 key prefix, e.g. 2025).")
    ap.add_argument("--mode", choices=("local", "lambda"), default="local",
                    help="Where to run the new prediction. Default: local.")
    ap.add_argument("--dest", default="s3://wri-restoration-geodata-ttc",
                    help="S3 bucket URI where ARD + FINAL.tif live. Default: s3://wri-restoration-geodata-ttc")
    ap.add_argument("--region", default="us-east-1",
                    help="AWS region for the bucket.")
    ap.add_argument(
        "--ard-variant", action="append", default=None, metavar="PREFIX:BASE_TMPL",
        help=(
            "Override the ARD candidate list. Repeat to provide multiple "
            "'PREFIX:BASE_TMPL' pairs in priority order. BASE_TMPL uses "
            "{year}/{x}/{y} placeholders. Example: "
            "--ard-variant 'dev-ttc-lithops-usw2/:{year}/{x}/{y}/raw'. "
            f"Default probes {len(ARD_KEY_VARIANTS)} built-in combinations: "
            f"{list(ARD_KEY_VARIANTS)}. "
            "FINAL.tif is always read from the bucket root."
        ),
    )
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"),
                    help="AWS profile (defaults to $AWS_PROFILE).")
    ap.add_argument("--model-dir", default=str(REPO_ROOT / "models"),
                    help="Directory containing predict_graph-172.pb + superresolve_graph.pb. "
                         "Only used in --mode local.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for overlap-calibration sampling. Default: 42 (matches local parity).")
    ap.add_argument("--env", default=os.environ.get("LITHOPS_ENV", "datalab-test"),
                    help="Lithops env dir under .lithops/. Used by --mode lambda. "
                         "Defaults to $LITHOPS_ENV or 'datalab-test'.")
    ap.add_argument("--memory-mb", type=int, default=6144,
                    help="Lambda memory (MB). Used by --mode lambda for cost + invocation. Default: 6144.")
    ap.add_argument("--scratch-prefix", default="_parity_scratch/",
                    help="S3 key prefix for the Lambda's scratch FINAL.tif "
                         "(deleted after the report is built). Default: _parity_scratch/")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="Don't delete the Lambda scratch key after the run (useful if debugging).")
    ap.add_argument("--output", default=None,
                    help="HTML output path. Default: temp/parity_<tile>_<year>[_lambda].html")
    args = ap.parse_args()

    x, y = _parse_tile(args.tile)
    output_path = Path(args.output) if args.output else (
        REPO_ROOT / "temp"
        / f"parity_{args.tile}_{args.year}{'_lambda' if args.mode == 'lambda' else ''}.html"
    )
    if args.mode == "local":
        model_dir = Path(args.model_dir)
        if not (model_dir / "predict_graph-172.pb").is_file():
            logger.error(
                f"Model dir missing predict_graph-172.pb: {model_dir}. "
                "Pass --model-dir or place models at repo-root/models/."
            )
            return 3

    logger.info(
        f"Tile parity: tile={args.tile} year={args.year} mode={args.mode} "
        f"dest={args.dest} seed={args.seed} output={output_path}"
    )

    from gri_tile_pipeline.storage.obstore_utils import from_dest

    store = from_dest(args.dest, region=args.region, profile=args.profile)

    # Prediction always at bucket root; ARD sources resolve per-source across
    # (prefix × base-pattern) combinations.
    _verify_prediction_exists(store, args.year, x, y)

    if args.ard_variant:
        variants: tuple[tuple[str, str], ...] = tuple(
            tuple(v.split(":", 1)) for v in args.ard_variant  # type: ignore[misc]
        )
        for v in variants:
            if len(v) != 2:
                raise SystemExit(
                    f"[CONFIG] --ard-variant must be 'PREFIX:BASE_TMPL', got: {v!r}"
                )
        resolved_keys = _resolve_ard_keys(store, args.year, x, y, variants=variants)
    else:
        resolved_keys = _resolve_ard_keys(store, args.year, x, y)

    existing = _load_existing_prediction(store, args.year, x, y)
    scratch_key: str | None = None

    try:
        if args.mode == "local":
            ard = _load_ard(store, resolved_keys)
            res = _run_local_predict(ard, Path(args.model_dir), args.seed)
            new_pred = res["pred"]
            execution = res["execution"]
        else:
            # Lambda mode — no local ARD load; the worker reads it via the
            # ard_keys_override map we just resolved.
            scratch_key = (
                f"{args.scratch_prefix.rstrip('/')}/"
                f"{args.year}/{x}/{y}/{args.tile}_FINAL_{uuid.uuid4().hex[:8]}.tif"
            )
            res = _invoke_lambda(
                args.year, x, y,
                dest=args.dest,
                resolved_keys=resolved_keys,
                scratch_key=scratch_key,
                seed=args.seed,
                env=args.env,
                memory_mb=args.memory_mb,
            )
            execution = res["execution"]
            new_pred = _download_scratch_prediction(store, scratch_key)

        stats = _build_report(
            args.tile, args.year, new_pred, existing, output_path, execution,
        )
    finally:
        if scratch_key and not args.keep_scratch:
            _delete_scratch(store, scratch_key)

    logger.info(
        f"Parity: pct_within_1={stats['pct_within_1']:.1f}% "
        f"pct_within_10={stats['pct_within_10']:.1f}% "
        f"corr={stats['correlation']:.4f} "
        f"corr_excl={stats['correlation_excl_outliers']:.4f} "
        f"max_diff={stats['max_abs_diff']:.0f} mean_diff={stats['mean_abs_diff']:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
