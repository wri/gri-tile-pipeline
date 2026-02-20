"""Run worker functions locally (in-process), bypassing Lithops/Lambda."""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

from loguru import logger

from gri_tile_pipeline.tracking import JobTracker, JobResult


def _invoke_worker(
    worker_fn: Callable,
    kwargs: Dict[str, Any],
    task_type: str,
    tracker: JobTracker,
) -> Dict[str, Any]:
    """Call *worker_fn* with *kwargs* and record the result in *tracker*."""
    tile_info = {
        k: kwargs[k] for k in ("year", "lon", "lat", "X_tile", "Y_tile") if k in kwargs
    }
    tile_key = f"{tile_info.get('X_tile')}X{tile_info.get('Y_tile')}Y"
    job_id = f"{task_type}_{tile_key}_{tile_info.get('year')}"

    logger.info(f"[local] Starting {task_type} for {tile_key}")
    t0 = time.perf_counter()

    try:
        result = worker_fn(**kwargs)
        duration = time.perf_counter() - t0

        if isinstance(result, dict):
            status = result.get("status", "success")
            tracker.add_result(JobResult(
                job_id=job_id,
                task_type=task_type,
                region="local",
                tile_info=tile_info,
                status=status,
                duration_sec=duration,
                result_data=result,
                error_message=result.get("error_message"),
                error_type=result.get("error_type"),
                quarters_succeeded=result.get("quarters_succeeded"),
                quarters_failed=result.get("quarters_failed"),
            ))
        else:
            tracker.add_result(JobResult(
                job_id=job_id,
                task_type=task_type,
                region="local",
                tile_info=tile_info,
                status="success",
                duration_sec=duration,
                result_data={"raw": str(result)},
            ))

        logger.info(f"[local] Completed {task_type} for {tile_key} in {duration:.1f}s")
        return result

    except Exception as exc:
        duration = time.perf_counter() - t0
        logger.error(f"[local] {task_type} failed for {tile_key}: {exc}")
        tracker.add_result(JobResult(
            job_id=job_id,
            task_type=task_type,
            region="local",
            tile_info=tile_info,
            status="failed",
            duration_sec=duration,
            error_message=str(exc),
            error_traceback=traceback.format_exc(),
            error_type="unknown",
        ))
        return {"status": "failed", "error_message": str(exc)}


def run_local_tasks(
    worker_fn: Callable,
    task_type: str,
    kwargs_list: List[Dict[str, Any]],
    tracker: JobTracker,
    max_workers: int = 1,
) -> None:
    """Call *worker_fn* for each kwargs dict, sequentially or in parallel.

    Args:
        worker_fn: The loader ``run()`` function to call.
        task_type: Label for reporting (e.g. ``"DEM"``, ``"S2"``).
        kwargs_list: One dict per tile with the worker's keyword args.
        tracker: Collects results for every invocation.
        max_workers: ``1`` for sequential (default, safe for memory-heavy
            workers), ``>1`` for thread-pool parallelism.
    """
    if not kwargs_list:
        return

    logger.info(f"[local] Running {len(kwargs_list)} {task_type} tasks (max_workers={max_workers})")

    if max_workers <= 1:
        for kw in kwargs_list:
            _invoke_worker(worker_fn, kw, task_type, tracker)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_invoke_worker, worker_fn, kw, task_type, tracker): kw
                for kw in kwargs_list
            }
            for fut in as_completed(futures):
                fut.result()  # propagate exceptions for logging (already caught inside)
