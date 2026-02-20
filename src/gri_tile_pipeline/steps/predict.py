"""Predict step: fan out tree cover prediction jobs via Lithops.

Orchestrates TF inference Lambda workers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import yaml
from loguru import logger

from gri_tile_pipeline.config import PipelineConfig
from gri_tile_pipeline.tiles.csv_io import read_tiles_csv
from gri_tile_pipeline.tracking import JobTracker
from gri_tile_pipeline.tracking.job_tracker import wait_all_with_tracking


# Historical average prediction duration (seconds)
AVG_PREDICT_DURATION = 180


# ------------------------------------
# Worker entry-point
# ------------------------------------

def _run_predict(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    from loaders.predict_tile import run
    return run(**kwargs)


# ------------------------------------
# Main orchestration
# ------------------------------------

def run_predict(
    tiles_csv: str,
    dest: str,
    cfg: PipelineConfig,
    *,
    model_path: str | None = None,
    memory_mb: int | None = None,
    retries: int | None = None,
    predict_cfg: str | None = None,
    skip_existing: bool = False,
    report_dir: str = "job_reports",
    debug: bool = False,
) -> JobTracker:
    """Fan out prediction jobs via Lithops.

    Returns the :class:`JobTracker` with all results.
    """
    import lithops
    from lithops import FunctionExecutor
    from lithops.retries import RetryingFuture

    runtime = cfg.predict.runtime
    memory_mb = memory_mb or cfg.predict.memory_mb
    retries = retries if retries is not None else cfg.predict.retries
    predict_cfg_path = predict_cfg or cfg.lithops.predict_config

    tiles = read_tiles_csv(tiles_csv)
    if not tiles:
        logger.warning("No tiles to process")
        return JobTracker(report_dir)

    if skip_existing:
        from gri_tile_pipeline.tiles.availability import filter_missing_tiles
        tiles = filter_missing_tiles(tiles, dest, check_type="predictions")
        if not tiles:
            logger.info("All prediction tiles already exist — nothing to do")
            return JobTracker(report_dir)

    tracker = JobTracker(report_dir)

    with open(predict_cfg_path) as f:
        lithops_cfg = yaml.safe_load(f)
    lithops_cfg.setdefault("aws_lambda", {})["runtime"] = runtime

    base_kwargs: List[Dict[str, Any]] = [
        {
            "year": t["year"],
            "lon": t["lon"],
            "lat": t["lat"],
            "X_tile": t["X_tile"],
            "Y_tile": t["Y_tile"],
            "dest": dest,
            "model_path": model_path,
            "debug": debug,
        }
        for t in tiles
    ]

    fexec = FunctionExecutor(config=lithops_cfg, runtime=runtime, runtime_memory=memory_mb)
    retry_exec = lithops.RetryingFunctionExecutor(fexec)

    futures: List[Tuple[RetryingFuture, str, str, Dict[str, Any]]] = []
    for kw in base_kwargs:
        tile_info = {k: kw[k] for k in ("year", "lon", "lat", "X_tile", "Y_tile")}
        futures.append((
            RetryingFuture(
                fexec.call_async(_run_predict, (kw,), include_modules=["loaders"]),
                _run_predict, (kw,), retries=retries,
            ),
            "PREDICT", "us-west-2", tile_info,
        ))

    logger.info(f"Submitted {len(futures)} prediction jobs")
    wait_all_with_tracking(retry_exec, futures, tracker)
    tracker.print_summary()
    tracker.save_reports()
    return tracker


# ------------------------------------
# Local execution
# ------------------------------------

def run_predict_local(
    tiles_csv: str,
    dest: str,
    cfg: PipelineConfig,
    *,
    model_path: str | None = None,
    skip_existing: bool = False,
    report_dir: str = "job_reports",
    debug: bool = False,
    max_workers: int = 1,
) -> JobTracker:
    """Run prediction locally — no Lithops.

    Returns the :class:`JobTracker` with all results.
    """
    from loaders.predict_tile import run as predict_run

    from gri_tile_pipeline.execution import run_local_tasks

    tiles = read_tiles_csv(tiles_csv)
    if not tiles:
        logger.warning("No tiles to process")
        return JobTracker(report_dir)

    if skip_existing:
        from gri_tile_pipeline.tiles.availability import filter_missing_tiles
        tiles = filter_missing_tiles(tiles, dest, check_type="predictions")
        if not tiles:
            logger.info("All prediction tiles already exist — nothing to do")
            return JobTracker(report_dir)

    tracker = JobTracker(report_dir)

    kwargs_list = [
        {
            "year": t["year"],
            "lon": t["lon"],
            "lat": t["lat"],
            "X_tile": t["X_tile"],
            "Y_tile": t["Y_tile"],
            "dest": dest,
            "model_path": model_path,
            "debug": debug,
        }
        for t in tiles
    ]

    run_local_tasks(predict_run, "PREDICT", kwargs_list, tracker, max_workers=max_workers)
    tracker.print_summary()
    tracker.save_reports()
    return tracker
