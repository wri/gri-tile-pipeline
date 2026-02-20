"""Download S1 RTC step: fan out S1 RTC acquisition via Lithops.

Wraps the logic from ``lithops_s1_rtc_orchestrator.py``.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from loguru import logger

from gri_tile_pipeline.config import PipelineConfig
from gri_tile_pipeline.tiles.csv_io import read_tiles_csv
from gri_tile_pipeline.tracking import JobTracker
from gri_tile_pipeline.tracking.job_tracker import wait_all_with_tracking


# ------------------------------------
# Worker entry-point
# ------------------------------------

def _run_s1_rtc(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """De-synchronised worker for S1 RTC acquisition."""
    time.sleep(random.uniform(0.0, 2.0))
    from loaders.download_s1_rtc import run
    return run(**kwargs)


# ------------------------------------
# Planetary Computer SAS token helpers
# ------------------------------------

def get_pc_collection_token(
    collection_id: str,
    api_key: str | None = None,
) -> dict:
    """Fetch a Planetary Computer SAS token for *collection_id*."""
    url = f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection_id}"
    headers = {}
    if api_key:
        headers["Ocp-Apim-Subscription-Key"] = api_key
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    expiry = None
    exp = data.get("msft:expiry") or data.get("expiry") or data.get("expires")
    if isinstance(exp, str):
        try:
            expiry = datetime.fromisoformat(exp.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            expiry = None
    return {"token": data["token"], "expiry": expiry}


def _load_token_cache(cache_path: str) -> dict | None:
    try:
        if not cache_path or not os.path.exists(cache_path):
            return None
        with open(cache_path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _save_token_cache(
    cache_path: str,
    collection_id: str,
    token: str,
    expiry: datetime | None,
) -> None:
    if not cache_path:
        return
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "collection_id": collection_id,
                    "token": token,
                    "expiry": expiry.isoformat() if isinstance(expiry, datetime) else None,
                },
                f,
                indent=2,
            )
    except Exception as e:
        logger.warning(f"Failed to write token cache to {cache_path}: {e}")


def ensure_pc_collection_token(
    collection_id: str,
    cache_path: str,
    api_key: str | None = None,
    min_ttl_seconds: int = 20 * 60,
) -> dict:
    """Return a valid SAS token, reusing cache if TTL is sufficient."""
    now = datetime.now(timezone.utc)
    cached = _load_token_cache(cache_path)
    if (
        cached
        and cached.get("collection_id") == collection_id
        and isinstance(cached.get("token"), str)
    ):
        exp_val = cached.get("expiry")
        if isinstance(exp_val, str) and exp_val.strip():
            try:
                expiry = datetime.fromisoformat(exp_val.replace("Z", "+00:00")).astimezone(timezone.utc)
                if (expiry - now).total_seconds() > min_ttl_seconds:
                    return {"token": cached["token"], "expiry": expiry, "source": "cache"}
            except Exception:
                pass

    token_info = get_pc_collection_token(collection_id, api_key=api_key)
    _save_token_cache(cache_path, collection_id, token_info["token"], token_info.get("expiry"))
    token_info["source"] = "fresh"
    return token_info


# ------------------------------------
# Main orchestration
# ------------------------------------

def run_download_s1_rtc(
    tiles_csv: str,
    dest: str,
    cfg: PipelineConfig,
    *,
    runtime: str | None = None,
    memory_mb: int | None = None,
    retries: int | None = None,
    s1_cfg: str | None = None,
    pc_token_cache: str = ".pc_sas_token_cache.json",
    pc_token_min_ttl_minutes: int = 20,
    report_dir: str = "job_reports",
    debug: bool = False,
    skip_existing: bool = False,
) -> JobTracker:
    """Fan out S1 RTC acquisition jobs via Lithops.

    Returns the :class:`JobTracker` with all results.
    """
    import lithops
    from lithops import FunctionExecutor
    from lithops.retries import RetryingFuture

    runtime = runtime or cfg.s1_rtc.runtime
    memory_mb = memory_mb or cfg.s1_rtc.memory_mb
    retries = retries if retries is not None else cfg.s1_rtc.retries
    s1_cfg_path = s1_cfg or cfg.lithops.s1_usw2_config

    tiles = read_tiles_csv(tiles_csv)
    if not tiles:
        logger.warning("No tiles to process")
        return JobTracker(report_dir)

    if skip_existing:
        from gri_tile_pipeline.tiles.availability import filter_missing_tiles
        tiles = filter_missing_tiles(tiles, dest, check_type="raw_ard")
        if not tiles:
            logger.info("All tiles already exist — nothing to do")
            return JobTracker(report_dir)

    # Fetch Planetary Computer SAS token
    pc_api_key = os.getenv("PC_SDK_SUBSCRIPTION_KEY") or os.getenv("PC_SUBSCRIPTION_KEY")
    token_info = ensure_pc_collection_token(
        "sentinel-1-rtc",
        cache_path=pc_token_cache,
        api_key=pc_api_key,
        min_ttl_seconds=pc_token_min_ttl_minutes * 60,
    )
    pc_sas_token = token_info["token"]
    logger.info(
        f"Using {token_info.get('source', 'unknown')} SAS token for sentinel-1-rtc"
    )

    tracker = JobTracker(report_dir)

    with open(s1_cfg_path) as f:
        cfg_usw2 = yaml.safe_load(f)
    cfg_usw2.setdefault("aws_lambda", {})["runtime"] = runtime

    base_kwargs: List[Dict[str, Any]] = [
        {
            "year": t["year"],
            "lon": t["lon"],
            "lat": t["lat"],
            "X_tile": t["X_tile"],
            "Y_tile": t["Y_tile"],
            "dest": dest,
            "sas_token": pc_sas_token,
            "debug": debug,
        }
        for t in tiles
    ]

    fexec = FunctionExecutor(config=cfg_usw2, runtime=runtime, runtime_memory=memory_mb)
    retry_exec = lithops.RetryingFunctionExecutor(fexec)

    futures: List[Tuple[RetryingFuture, str, str, Dict[str, Any]]] = []
    for kw in base_kwargs:
        tile_info = {k: kw[k] for k in ("year", "lon", "lat", "X_tile", "Y_tile")}
        futures.append((
            RetryingFuture(
                fexec.call_async(_run_s1_rtc, (kw,), include_modules=["loaders"]),
                _run_s1_rtc, (kw,), retries=retries,
            ),
            "S1_RTC", "us-west-2", tile_info,
        ))

    logger.info(f"Submitted {len(futures)} S1 RTC jobs to us-west-2")
    wait_all_with_tracking(retry_exec, futures, tracker)
    tracker.print_summary()
    tracker.save_reports()
    return tracker


# ------------------------------------
# Local execution
# ------------------------------------

def run_download_s1_rtc_local(
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
    """Download S1 RTC locally — no Lithops.

    Returns the :class:`JobTracker` with all results.
    """
    from loaders.download_s1_rtc import run as s1_rtc_run

    from gri_tile_pipeline.execution import run_local_tasks

    tiles = read_tiles_csv(tiles_csv)
    if not tiles:
        logger.warning("No tiles to process")
        return JobTracker(report_dir)

    if skip_existing:
        from gri_tile_pipeline.tiles.availability import filter_missing_tiles
        tiles = filter_missing_tiles(tiles, dest, check_type="raw_ard")
        if not tiles:
            logger.info("All tiles already exist — nothing to do")
            return JobTracker(report_dir)

    # Fetch Planetary Computer SAS token
    pc_api_key = os.getenv("PC_SDK_SUBSCRIPTION_KEY") or os.getenv("PC_SUBSCRIPTION_KEY")
    token_info = ensure_pc_collection_token(
        "sentinel-1-rtc",
        cache_path=pc_token_cache,
        api_key=pc_api_key,
        min_ttl_seconds=pc_token_min_ttl_minutes * 60,
    )
    pc_sas_token = token_info["token"]
    logger.info(
        f"Using {token_info.get('source', 'unknown')} SAS token for sentinel-1-rtc"
    )

    tracker = JobTracker(report_dir)

    kwargs_list = [
        {
            "year": t["year"],
            "lon": t["lon"],
            "lat": t["lat"],
            "X_tile": t["X_tile"],
            "Y_tile": t["Y_tile"],
            "dest": dest,
            "sas_token": pc_sas_token,
            "debug": debug,
        }
        for t in tiles
    ]

    run_local_tasks(s1_rtc_run, "S1_RTC", kwargs_list, tracker, max_workers=max_workers)
    tracker.print_summary()
    tracker.save_reports()
    return tracker
