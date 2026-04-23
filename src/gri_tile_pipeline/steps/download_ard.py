"""Download ARD step: fan out DEM + S1 RTC + S2 via Lithops.

Wraps the logic from ``lithops_job_tracker.py``.

S1 uses the Planetary Computer RTC (radiometric terrain corrected) source
by default.  The legacy Earth Search GRD-based S1 loader is available via
``run_download_ard_legacy_s1_local``.
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
# Worker entry-points for Lithops (importable standalone module)
# ------------------------------------
# Workers live in lithops_workers.py so Lithops can pickle/unpickle them
# on Lambda without needing gri_tile_pipeline installed.
# The path import ensures loaders/ and lithops_workers are findable.

import gri_tile_pipeline.steps._lithops_path  # noqa: F401  (adds repo root to sys.path)

from lithops_workers import run_dem as _run_dem
from lithops_workers import run_s1_rtc as _run_s1_rtc
from lithops_workers import run_s1_legacy as _run_s1_legacy
from lithops_workers import run_s2 as _run_s2


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
    s1_cfg: str | None = None,
    pc_token_cache: str = ".pc_sas_token_cache.json",
    pc_token_min_ttl_minutes: int = 20,
    report_dir: str = "job_reports",
    debug: bool = False,
    skip_existing: bool = False,
) -> JobTracker:
    """Fan out DEM (eu-central-1) + S1 RTC + S2 (us-west-2) downloads.

    S1 uses Planetary Computer RTC by default.

    Returns the :class:`JobTracker` with all results.
    """
    import lithops
    from lithops import FunctionExecutor
    from lithops.retries import RetryingFuture

    from gri_tile_pipeline.steps.download_s1_rtc import ensure_pc_collection_token

    runtime = runtime or cfg.download.runtime
    memory_mb = memory_mb or cfg.download.memory_mb
    retries = retries if retries is not None else cfg.download.retries
    euc1_cfg_path = euc1_cfg or cfg.lithops.euc1_config
    usw2_cfg_path = usw2_cfg or cfg.lithops.usw2_config
    s1_cfg_path = s1_cfg or cfg.lithops.s1_usw2_config

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

    # Fetch Planetary Computer SAS token for S1 RTC
    pc_api_key = os.getenv("PC_SDK_SUBSCRIPTION_KEY") or os.getenv("PC_SUBSCRIPTION_KEY")
    token_info = ensure_pc_collection_token(
        "sentinel-1-rtc",
        cache_path=pc_token_cache,
        api_key=pc_api_key,
        min_ttl_seconds=pc_token_min_ttl_minutes * 60,
    )
    pc_sas_token = token_info["token"]
    logger.info(f"Using {token_info.get('source', 'unknown')} SAS token for S1 RTC")

    tracker = JobTracker(report_dir)

    # Load per-region configs
    with open(euc1_cfg_path) as f:
        cfg_euc1 = yaml.safe_load(f)
    with open(usw2_cfg_path) as f:
        cfg_usw2 = yaml.safe_load(f)
    with open(s1_cfg_path) as f:
        cfg_s1 = yaml.safe_load(f)

    cfg_euc1.setdefault("aws_lambda", {})["runtime"] = runtime
    cfg_usw2.setdefault("aws_lambda", {})["runtime"] = runtime
    cfg_s1.setdefault("aws_lambda", {})["runtime"] = cfg.s1_rtc.runtime

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
    fexec_s1 = FunctionExecutor(
        config=cfg_s1, runtime=cfg.s1_rtc.runtime,
        runtime_memory=cfg.s1_rtc.memory_mb,
    )

    retry_euc1 = lithops.RetryingFunctionExecutor(fexec_euc1)
    retry_usw2 = lithops.RetryingFunctionExecutor(fexec_usw2)
    retry_s1 = lithops.RetryingFunctionExecutor(fexec_s1)

    # DEM -> eu-central-1
    futures_euc1: List[Tuple[RetryingFuture, str, str, Dict[str, Any]]] = []
    for kw in base_kwargs:
        tile_info = {k: kw[k] for k in ("year", "lon", "lat", "X_tile", "Y_tile")}
        futures_euc1.append((
            RetryingFuture(
                fexec_euc1.call_async(_run_dem, (kw,), include_modules=["loaders", "lithops_workers"]),
                _run_dem, (kw,), retries=retries,
            ),
            "DEM", "eu-central-1", tile_info,
        ))

    # S1 RTC -> us-west-2 (Planetary Computer)
    s1_retries = retries if retries is not None else cfg.s1_rtc.retries
    futures_s1: List[Tuple[RetryingFuture, str, str, Dict[str, Any]]] = []
    for kw in base_kwargs:
        tile_info = {k: kw[k] for k in ("year", "lon", "lat", "X_tile", "Y_tile")}
        s1_kw = {**kw, "sas_token": pc_sas_token}
        futures_s1.append((
            RetryingFuture(
                fexec_s1.call_async(_run_s1_rtc, (s1_kw,), include_modules=["loaders", "lithops_workers"]),
                _run_s1_rtc, (s1_kw,), retries=s1_retries,
            ),
            "S1_RTC", "us-west-2", tile_info,
        ))

    # S2 -> us-west-2
    futures_usw2: List[Tuple[RetryingFuture, str, str, Dict[str, Any]]] = []
    for kw in base_kwargs:
        tile_info = {k: kw[k] for k in ("year", "lon", "lat", "X_tile", "Y_tile")}
        futures_usw2.append((
            RetryingFuture(
                fexec_usw2.call_async(_run_s2, (kw,), include_modules=["loaders", "lithops_workers"]),
                _run_s2, (kw,), retries=retries,
            ),
            "S2", "us-west-2", tile_info,
        ))

    total = len(futures_euc1) + len(futures_s1) + len(futures_usw2)
    logger.info(
        f"Submitted {total} jobs: {len(futures_euc1)} eu-central-1 (DEM), "
        f"{len(futures_s1) + len(futures_usw2)} us-west-2 (S1 RTC + S2)"
    )

    if futures_euc1:
        wait_all_with_tracking(retry_euc1, futures_euc1, tracker)
    if futures_s1:
        wait_all_with_tracking(retry_s1, futures_s1, tracker)
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
    pc_token_cache: str = ".pc_sas_token_cache.json",
    pc_token_min_ttl_minutes: int = 20,
    report_dir: str = "job_reports",
    debug: bool = False,
    skip_existing: bool = False,
    max_workers: int = 1,
) -> JobTracker:
    """Download DEM + S1 RTC + S2 ARD locally — no Lithops.

    S1 uses Planetary Computer RTC by default.

    Returns the :class:`JobTracker` with all results.
    """
    from loaders.download_dem import run as dem_run
    from loaders.download_s1_rtc import run as s1_rtc_run
    from loaders.download_s2 import run as s2_run

    from gri_tile_pipeline.execution import run_local_tasks
    from gri_tile_pipeline.steps.download_s1_rtc import ensure_pc_collection_token

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

    # Fetch Planetary Computer SAS token for S1 RTC
    pc_api_key = os.getenv("PC_SDK_SUBSCRIPTION_KEY") or os.getenv("PC_SUBSCRIPTION_KEY")
    token_info = ensure_pc_collection_token(
        "sentinel-1-rtc",
        cache_path=pc_token_cache,
        api_key=pc_api_key,
        min_ttl_seconds=pc_token_min_ttl_minutes * 60,
    )
    pc_sas_token = token_info["token"]
    logger.info(f"Using {token_info.get('source', 'unknown')} SAS token for S1 RTC")

    tracker = JobTracker(report_dir)
    kwargs_list = _build_base_kwargs(tiles, dest, debug)
    s1_kwargs_list = [{**kw, "sas_token": pc_sas_token} for kw in kwargs_list]

    # DEM is lightweight — safe to parallelize
    run_local_tasks(dem_run, "DEM", kwargs_list, tracker, max_workers=max_workers)
    # S1 RTC and S2 are memory-heavy — run sequentially by default
    run_local_tasks(s1_rtc_run, "S1_RTC", s1_kwargs_list, tracker)
    run_local_tasks(s2_run, "S2", kwargs_list, tracker)

    tracker.print_summary()
    tracker.save_reports()
    return tracker


def run_download_ard_legacy_s1_local(
    tiles_csv: str,
    dest: str,
    cfg: PipelineConfig,
    *,
    report_dir: str = "job_reports",
    debug: bool = False,
    skip_existing: bool = False,
    max_workers: int = 1,
) -> JobTracker:
    """Download DEM + S1 (legacy Earth Search GRD) + S2 locally.

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

    run_local_tasks(dem_run, "DEM", kwargs_list, tracker, max_workers=max_workers)
    run_local_tasks(s1_run, "S1", kwargs_list, tracker)
    run_local_tasks(s2_run, "S2", kwargs_list, tracker)

    tracker.print_summary()
    tracker.save_reports()
    return tracker
