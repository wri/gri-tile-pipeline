"""Download ARD step: fan out DEM + S1 + S2 via Lithops.

Wraps the logic from ``lithops_job_tracker.py``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

import yaml
from loguru import logger

from gri_tile_pipeline.config import PipelineConfig
from gri_tile_pipeline.tiles.csv_io import read_tiles_csv
from gri_tile_pipeline.tracking import JobTracker, JobResult
from gri_tile_pipeline.tracking.job_tracker import wait_all_with_tracking


# ------------------------------------
# Worker entry-points (top-level for pickling)
# ------------------------------------

def _run_dem(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    from loaders.download_dem import run
    return run(**kwargs)


def _run_s1(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    from loaders.download_s1 import run
    return run(**kwargs)


def _run_s2(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    from loaders.download_s2 import run
    return run(**kwargs)


# ------------------------------------
# Cost estimation
# ------------------------------------

PRICE_PER_GB_SEC = 0.00001667

# Historical average durations per task type (seconds)
AVG_DURATIONS = {"DEM": 9, "S1": 14, "S2": 62}


def estimate_cost(num_tiles: int, memory_mb: int) -> Dict[str, float]:
    """Return per-task-type and total estimated Lambda costs."""
    mem_gb = memory_mb / 1024.0
    costs = {}
    for task, avg_sec in AVG_DURATIONS.items():
        costs[task] = num_tiles * avg_sec * mem_gb * PRICE_PER_GB_SEC
    costs["total"] = sum(costs.values())
    return costs


# ------------------------------------
# Main orchestration
# ------------------------------------

def run_download_ard(
    tiles_csv: str,
    dest: str,
    cfg: PipelineConfig,
    *,
    runtime: str | None = None,
    memory_mb: int | None = None,
    retries: int | None = None,
    euc1_cfg: str | None = None,
    usw2_cfg: str | None = None,
    report_dir: str = "job_reports",
    debug: bool = False,
    skip_existing: bool = False,
) -> JobTracker:
    """Fan out DEM + S1 (eu-central-1) and S2 (us-west-2) downloads.

    Returns the :class:`JobTracker` with all results.
    """
    import lithops
    from lithops import FunctionExecutor
    from lithops.retries import RetryingFuture

    runtime = runtime or cfg.download.runtime
    memory_mb = memory_mb or cfg.download.memory_mb
    retries = retries if retries is not None else cfg.download.retries
    euc1_cfg_path = euc1_cfg or cfg.lithops.euc1_config
    usw2_cfg_path = usw2_cfg or cfg.lithops.usw2_config

    tiles = read_tiles_csv(tiles_csv)
    if not tiles:
        logger.warning("No tiles to process")
        return JobTracker(report_dir)

    if skip_existing:
        from gri_tile_pipeline.tiles.availability import filter_missing_tiles
        tiles = filter_missing_tiles(tiles, dest, check_type="raw_ard")
        if not tiles:
            logger.info("All tiles already exist on destination — nothing to do")
            return JobTracker(report_dir)

    tracker = JobTracker(report_dir)

    # Load per-region configs
    with open(euc1_cfg_path) as f:
        cfg_euc1 = yaml.safe_load(f)
    with open(usw2_cfg_path) as f:
        cfg_usw2 = yaml.safe_load(f)

    cfg_euc1.setdefault("aws_lambda", {})["runtime"] = runtime
    cfg_usw2.setdefault("aws_lambda", {})["runtime"] = runtime

    base_kwargs: List[Dict[str, Any]] = [
        {
            "year": t["year"],
            "lon": t["lon"],
            "lat": t["lat"],
            "X_tile": t["X_tile"],
            "Y_tile": t["Y_tile"],
            "dest": dest,
            "debug": debug,
        }
        for t in tiles
    ]

    fexec_euc1 = FunctionExecutor(config=cfg_euc1, runtime=runtime, runtime_memory=memory_mb)
    fexec_usw2 = FunctionExecutor(config=cfg_usw2, runtime=runtime, runtime_memory=memory_mb)

    retry_euc1 = lithops.RetryingFunctionExecutor(fexec_euc1)
    retry_usw2 = lithops.RetryingFunctionExecutor(fexec_usw2)

    # DEM + S1 -> eu-central-1
    futures_euc1: List[Tuple[RetryingFuture, str, str, Dict[str, Any]]] = []
    for kw in base_kwargs:
        tile_info = {k: kw[k] for k in ("year", "lon", "lat", "X_tile", "Y_tile")}
        futures_euc1.append((
            RetryingFuture(
                fexec_euc1.call_async(_run_dem, (kw,), include_modules=["loaders"]),
                _run_dem, (kw,), retries=retries,
            ),
            "DEM", "eu-central-1", tile_info,
        ))
        futures_euc1.append((
            RetryingFuture(
                fexec_euc1.call_async(_run_s1, (kw,), include_modules=["loaders"]),
                _run_s1, (kw,), retries=retries,
            ),
            "S1", "eu-central-1", tile_info,
        ))

    # S2 -> us-west-2
    futures_usw2: List[Tuple[RetryingFuture, str, str, Dict[str, Any]]] = []
    for kw in base_kwargs:
        tile_info = {k: kw[k] for k in ("year", "lon", "lat", "X_tile", "Y_tile")}
        futures_usw2.append((
            RetryingFuture(
                fexec_usw2.call_async(_run_s2, (kw,), include_modules=["loaders"]),
                _run_s2, (kw,), retries=retries,
            ),
            "S2", "us-west-2", tile_info,
        ))

    total = len(futures_euc1) + len(futures_usw2)
    logger.info(
        f"Submitted {total} jobs: {len(futures_euc1)} eu-central-1, "
        f"{len(futures_usw2)} us-west-2"
    )

    if futures_euc1:
        wait_all_with_tracking(retry_euc1, futures_euc1, tracker)
    if futures_usw2:
        wait_all_with_tracking(retry_usw2, futures_usw2, tracker)

    tracker.print_summary()
    tracker.save_reports()
    return tracker


# ------------------------------------
# Local execution
# ------------------------------------

def _build_base_kwargs(tiles, dest, debug=False):
    """Build list of per-tile keyword dicts for workers."""
    return [
        {
            "year": t["year"],
            "lon": t["lon"],
            "lat": t["lat"],
            "X_tile": t["X_tile"],
            "Y_tile": t["Y_tile"],
            "dest": dest,
            "debug": debug,
        }
        for t in tiles
    ]


def run_download_ard_local(
    tiles_csv: str,
    dest: str,
    cfg: PipelineConfig,
    *,
    report_dir: str = "job_reports",
    debug: bool = False,
    skip_existing: bool = False,
    max_workers: int = 1,
) -> JobTracker:
    """Download DEM + S1 + S2 ARD locally — no Lithops.

    Returns the :class:`JobTracker` with all results.
    """
    from loaders.download_dem import run as dem_run
    from loaders.download_s1 import run as s1_run
    from loaders.download_s2 import run as s2_run

    from gri_tile_pipeline.execution import run_local_tasks

    tiles = read_tiles_csv(tiles_csv)
    if not tiles:
        logger.warning("No tiles to process")
        return JobTracker(report_dir)

    if skip_existing:
        from gri_tile_pipeline.tiles.availability import filter_missing_tiles
        tiles = filter_missing_tiles(tiles, dest, check_type="raw_ard")
        if not tiles:
            logger.info("All tiles already exist on destination — nothing to do")
            return JobTracker(report_dir)

    tracker = JobTracker(report_dir)
    kwargs_list = _build_base_kwargs(tiles, dest, debug)

    # DEM is lightweight — safe to parallelize
    run_local_tasks(dem_run, "DEM", kwargs_list, tracker, max_workers=max_workers)
    # S1 and S2 are memory-heavy — run sequentially by default
    run_local_tasks(s1_run, "S1", kwargs_list, tracker)
    run_local_tasks(s2_run, "S2", kwargs_list, tracker)

    tracker.print_summary()
    tracker.save_reports()
    return tracker
