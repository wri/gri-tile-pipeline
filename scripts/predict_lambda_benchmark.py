#!/usr/bin/env python3
"""Gate C: benchmark the deployed predict Lambda.

Submits N tile predictions through the same FunctionExecutor path the
pipeline uses, then reports p50/p95/p99 wallclock, cold-start count,
throughput at configured concurrency, and cross-region-byte expectation.

Use after every ``make -C infra build-all ENV=land-research`` (and any
region or memory change to the runtime) to catch regressions and quantify
whether Lambda still fits the workload.

Usage (from repo root, after ``aws sso login --profile resto-user``)::

    AWS_PROFILE=resto-user LITHOPS_ENV=land-research \\
        uv run python scripts/predict_lambda_benchmark.py --tiles 20

Outputs:
    benchmarks/<UTC-date>-<git-sha>.csv   per-tile timings
    stdout                                summary table

Exit codes:
    0 — all invocations succeeded and benchmarks written
    2 — ARD missing on S3 for one or more selected tiles (precondition)
    3 — one or more Lambda invocations failed
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent

# Re-use the known-tile dictionary from the smoke script to avoid duplication.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from predict_lambda_smoke import KNOWN_TILES, _check_ard  # noqa: E402

TTC_BUCKET_REGION = "us-east-1"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "nogit"


def _load_lithops_cfg(env: str) -> tuple[dict, str]:
    cfg_path = REPO_ROOT / ".lithops" / env / "config.predict.yaml"
    if not cfg_path.is_file():
        sys.exit(f"[bench] Lithops config missing: {cfg_path}. Run 'make -C infra render ENV={env}'.")
    cfg = yaml.safe_load(cfg_path.read_text())
    lambda_region = cfg.get("aws_lambda", {}).get("region") or cfg.get("aws", {}).get("region")
    return cfg, lambda_region or "unknown"


def _pick_tiles(n: int, extra_labels: list[str] | None) -> list[dict]:
    labels = list(KNOWN_TILES.keys())
    if extra_labels:
        labels = extra_labels + [l for l in labels if l not in extra_labels]
    # Cycle through the pool if n exceeds unique tiles — intentional: repeats
    # force warm starts so we observe both cold and warm invocation cost.
    tiles: list[dict] = []
    for i in range(n):
        label = labels[i % len(labels)]
        tile = dict(KNOWN_TILES[label])
        tile["_label"] = label
        tile["_slot"] = i
        tiles.append(tile)
    return tiles


def _submit_and_time(
    lithops_cfg: dict,
    tiles: list[dict],
    year: int,
    dest: str,
    memory_mb: int,
    timeout: int,
    runtime: str,
) -> list[dict]:
    import importlib
    import lithops
    from lithops import FunctionExecutor

    # Both paths are required: src/ for `lithops_workers`, repo root for
    # `loaders` (top-level package not under src/). Matches predict_lambda_smoke.py.
    repo_root = str(REPO_ROOT)
    src_dir = str(REPO_ROOT / "src")
    for p in (repo_root, src_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    lithops_workers = importlib.import_module("lithops_workers")

    lithops_cfg.setdefault("aws_lambda", {})["runtime"] = runtime

    fexec = FunctionExecutor(config=lithops_cfg, runtime=runtime, runtime_memory=memory_mb)
    submissions: list[tuple[dict, float, object]] = []

    batch_start = time.time()
    for tile in tiles:
        kwargs = {
            "year": year,
            "lon": tile["lon"],
            "lat": tile["lat"],
            "X_tile": tile["X_tile"],
            "Y_tile": tile["Y_tile"],
            "dest": dest,
            "model_path": None,
            "debug": False,
        }
        t_sub = time.time()
        future = fexec.call_async(
            lithops_workers.run_predict, (kwargs,),
            include_modules=["loaders", "lithops_workers"],
        )
        submissions.append((tile, t_sub, future))

    rows: list[dict] = []
    for tile, t_sub, future in submissions:
        status = "ok"
        lambda_duration = None
        err = ""
        t_done_rel = None
        phase_timings: dict[str, float] = {}
        worker_wallclock: float | None = None
        try:
            # Lithops 3.6.1 moved the timeout kwarg off ResponseFuture.result
            # onto the executor. get_result expects a list-like fs (it calls
            # len() on it). Same pattern used by predict_lambda_smoke.py.
            results = fexec.get_result([future], timeout=timeout)
            # get_result may return a scalar (single future) or list.
            result = results[0] if isinstance(results, list) else results
            t_done = time.time()
            t_done_rel = t_done - batch_start
            stats = getattr(future, "stats", {}) or {}
            # Lithops names vary across versions; try a few.
            for k in ("worker_exec_time", "worker_func_exec_time", "host_job_execution_time"):
                if k in stats:
                    lambda_duration = float(stats[k])
                    break
            # Phase timings come from the Lambda return dict (see
            # loaders/predict_tile.py::run). Only present when the worker
            # completed without raising.
            if isinstance(result, dict):
                pt = result.get("phase_timings") or {}
                if isinstance(pt, dict):
                    phase_timings = {str(k): float(v) for k, v in pt.items()}
                wc = result.get("wallclock_sec")
                if wc is not None:
                    worker_wallclock = float(wc)
        except Exception as e:
            status = "error"
            err = str(e)[:200]
            t_done = time.time()
            t_done_rel = t_done - batch_start

        rows.append({
            "slot": tile["_slot"],
            "tile": tile["_label"],
            "status": status,
            "submit_s": round(t_sub - batch_start, 3),
            "complete_s": round(t_done_rel or 0.0, 3),
            "wallclock_s": round((t_done_rel or 0.0) - (t_sub - batch_start), 3),
            "lambda_duration_s": round(lambda_duration, 3) if lambda_duration else "",
            "worker_wallclock_s": round(worker_wallclock, 3) if worker_wallclock else "",
            "phase_timings": phase_timings,
            "error": err,
        })

    return rows


def _summarize(rows: list[dict], lambda_region: str, max_workers: int, memory_mb: int) -> None:
    ok = [r for r in rows if r["status"] == "ok"]
    wall = [r["wallclock_s"] for r in ok]
    if not wall:
        print("[bench] All invocations failed. Cannot summarize.")
        return

    # Cold-start heuristic: first invocation per "concurrent slot" tends to be
    # cold. With max_workers ≥ N we treat the first N_unique invocations as
    # cold. This is an approximation — Lithops/Lambda don't surface cold-start
    # status reliably, but the first few wallclocks being much larger than the
    # trailing ones is a strong signal.
    n_cold_estimate = min(max_workers, len(rows))
    first_batch = sorted(wall)[:n_cold_estimate]
    warm_batch = sorted(wall)[n_cold_estimate:]

    total_elapsed = max(r["complete_s"] for r in ok)
    throughput = len(ok) / (total_elapsed / 60) if total_elapsed else 0.0

    cross_region = lambda_region != TTC_BUCKET_REGION

    def _p(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
        return s[k]

    print()
    print("[bench] Summary")
    print(f"    invocations:         {len(rows)} ({len(ok)} ok, {len(rows) - len(ok)} failed)")
    print(f"    lambda region:       {lambda_region}")
    print(f"    ttc bucket region:   {TTC_BUCKET_REGION}")
    print(f"    cross-region egress: {'YES (paying for it)' if cross_region else 'no (co-located)'}")
    print(f"    runtime memory:      {memory_mb} MB")
    print(f"    max workers:         {max_workers}")
    print()
    print(f"    wallclock p50:       {_p(wall, 0.50):.1f} s")
    print(f"    wallclock p95:       {_p(wall, 0.95):.1f} s")
    print(f"    wallclock p99:       {_p(wall, 0.99):.1f} s")
    print(f"    wallclock mean:      {statistics.mean(wall):.1f} s")
    if len(wall) >= 2:
        print(f"    wallclock stdev:     {statistics.stdev(wall):.1f} s")
    print()
    if first_batch and warm_batch:
        print(f"    cold-est median:     {statistics.median(first_batch):.1f} s (first {len(first_batch)} invocations)")
        print(f"    warm-est median:     {statistics.median(warm_batch):.1f} s (remaining {len(warm_batch)} invocations)")
    print()
    print(f"    total batch elapsed: {total_elapsed:.1f} s")
    print(f"    throughput:          {throughput:.1f} tiles/min")

    # Per-phase rollup. Only tiles that returned phase_timings contribute.
    phase_rows = [r for r in ok if r.get("phase_timings")]
    if phase_rows:
        phases: dict[str, list[float]] = {}
        for r in phase_rows:
            for name, sec in r["phase_timings"].items():
                phases.setdefault(name, []).append(float(sec))
        worker_wall = [
            float(r["worker_wallclock_s"])
            for r in phase_rows
            if r.get("worker_wallclock_s") not in (None, "")
        ]
        denom = statistics.mean(worker_wall) if worker_wall else None
        print()
        print(f"[bench] Per-phase timings (n={len(phase_rows)} tiles with phase data)")
        header = f"    {'phase':<30s} {'p50 (s)':>10s} {'p95 (s)':>10s} {'mean (s)':>10s}"
        if denom:
            header += f" {'% of wall':>10s}"
        print(header)
        for name in sorted(phases, key=lambda n: -statistics.mean(phases[n])):
            vals = phases[name]
            line = (
                f"    {name:<30s} "
                f"{_p(vals, 0.50):>10.2f} "
                f"{_p(vals, 0.95):>10.2f} "
                f"{statistics.mean(vals):>10.2f}"
            )
            if denom:
                line += f" {100 * statistics.mean(vals) / denom:>9.1f}%"
            print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default=os.environ.get("LITHOPS_ENV", "land-research"))
    parser.add_argument("--tiles", type=int, default=20, help="Number of invocations to submit")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--dest", default="s3://wri-restoration-geodata-ttc")
    parser.add_argument("--memory-mb", type=int, default=6144)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--runtime", default="ttc-predict-dev")
    parser.add_argument("--extra-tile", action="append", default=[],
                        help="Extra tile labels to include ahead of the default pool (repeatable).")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "benchmarks")
    args = parser.parse_args()

    lithops_cfg, lambda_region = _load_lithops_cfg(args.env)
    max_workers = int(lithops_cfg.get("aws_lambda", {}).get("max_workers", 1))

    tiles = _pick_tiles(args.tiles, args.extra_tile)
    logger.info(f"Selected {len(tiles)} tiles. Lambda region={lambda_region}. Destination={args.dest}.")

    # Precondition: ARD must exist for every unique tile in the batch.
    seen = set()
    for t in tiles:
        if t["_label"] in seen:
            continue
        seen.add(t["_label"])
        try:
            _check_ard(t, args.year, args.dest)
        except SystemExit as e:
            logger.error(str(e))
            return 2

    rows = _submit_and_time(
        lithops_cfg, tiles, args.year, args.dest,
        args.memory_mb, args.timeout, args.runtime,
    )
    n_failed = sum(1 for r in rows if r["status"] != "ok")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = _git_sha()

    # Top-level CSV: one row per invocation, excluding the nested phase dict.
    out_csv = args.out_dir / f"{stamp}-{sha}.csv"
    top_fields = [k for k in rows[0].keys() if k != "phase_timings"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=top_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: v for k, v in r.items() if k != "phase_timings"})
    logger.info(f"Wrote per-invocation timings to {out_csv}")

    # Phase CSV: one row per invocation, one column per phase observed in the
    # batch. Only emitted when at least one tile returned phase_timings.
    phase_rows = [r for r in rows if r.get("phase_timings")]
    if phase_rows:
        all_phases = sorted({p for r in phase_rows for p in r["phase_timings"]})
        phase_csv = args.out_dir / f"{stamp}-{sha}-phases.csv"
        with phase_csv.open("w", newline="") as f:
            fieldnames = ["slot", "tile", "worker_wallclock_s"] + all_phases
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in phase_rows:
                row = {
                    "slot": r["slot"],
                    "tile": r["tile"],
                    "worker_wallclock_s": r["worker_wallclock_s"],
                }
                for p in all_phases:
                    row[p] = r["phase_timings"].get(p)
                writer.writerow(row)
        logger.info(f"Wrote per-phase timings to {phase_csv}")

    _summarize(rows, lambda_region, max_workers, args.memory_mb)

    return 3 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
