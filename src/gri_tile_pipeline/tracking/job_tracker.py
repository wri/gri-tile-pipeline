"""Unified JobTracker with multi-format report generation."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from gri_tile_pipeline.tracking.job_result import JobResult


class JobTracker:
    """Centralized job tracking and reporting."""

    def __init__(self, output_dir: str = "job_reports"):
        self.output_dir = output_dir
        self.results: List[JobResult] = []
        self.start_time = datetime.now()
        os.makedirs(output_dir, exist_ok=True)

    def add_result(self, result: JobResult) -> None:
        self.results.append(result)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def save_reports(self) -> None:
        """Save JSON, CSV, text, and failed-jobs reports."""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

        # 1. Detailed JSON
        json_path = os.path.join(self.output_dir, f"job_report_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2, default=str)

        # 2. CSV summary
        csv_path = os.path.join(self.output_dir, f"job_summary_{timestamp}.csv")
        self._save_csv_summary(csv_path)

        # 3. Human-readable text
        txt_path = os.path.join(self.output_dir, f"job_report_{timestamp}.txt")
        self._save_text_report(txt_path)

        # 4. Failed jobs only
        failed = [r for r in self.results if r.status not in ("success",)]
        if failed:
            failed_path = os.path.join(
                self.output_dir, f"failed_jobs_{timestamp}.json"
            )
            with open(failed_path, "w") as f:
                json.dump([r.to_dict() for r in failed], f, indent=2, default=str)

        logger.info(f"Reports saved to {self.output_dir}/")

    def _save_csv_summary(self, path: str) -> None:
        fieldnames = [
            "job_id", "task_type", "region", "status",
            "year", "lon", "lat", "X_tile", "Y_tile",
            "duration_sec", "error_type", "total_retry_count",
            "quarters_succeeded", "quarters_failed", "error_message",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow({
                    "job_id": r.job_id,
                    "task_type": r.task_type,
                    "region": r.region,
                    "status": r.status,
                    "year": r.tile_info.get("year"),
                    "lon": r.tile_info.get("lon"),
                    "lat": r.tile_info.get("lat"),
                    "X_tile": r.tile_info.get("X_tile"),
                    "Y_tile": r.tile_info.get("Y_tile"),
                    "duration_sec": r.duration_sec,
                    "error_type": r.error_type,
                    "total_retry_count": r.total_retry_count,
                    "quarters_succeeded": (
                        ",".join(r.quarters_succeeded) if r.quarters_succeeded else None
                    ),
                    "quarters_failed": (
                        ",".join(r.quarters_failed) if r.quarters_failed else None
                    ),
                    "error_message": (
                        r.error_message[:100] if r.error_message else None
                    ),
                })

    def _save_text_report(self, path: str) -> None:
        total = len(self.results)
        if total == 0:
            with open(path, "w") as f:
                f.write("No jobs were executed.\n")
            return

        succeeded = sum(1 for r in self.results if r.status == "success")
        partial = sum(1 for r in self.results if r.status == "partial")
        failed = sum(1 for r in self.results if r.status == "failed")
        errored = sum(1 for r in self.results if r.status == "error")
        infra_errored = sum(1 for r in self.results if r.status == "infra_error")

        by_type: Dict[str, Dict[str, int]] = {}
        for r in self.results:
            bucket = by_type.setdefault(
                r.task_type,
                {"success": 0, "partial": 0, "failed": 0, "error": 0, "infra_error": 0},
            )
            if r.status in bucket:
                bucket[r.status] += 1

        durations = [r.duration_sec for r in self.results if r.duration_sec]
        avg_dur = sum(durations) / len(durations) if durations else 0
        max_dur = max(durations) if durations else 0
        min_dur = min(durations) if durations else 0

        with open(path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("LITHOPS JOB EXECUTION REPORT\n")
            f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write("=" * 60 + "\n\n")

            f.write("OVERALL SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Jobs:        {total}\n")
            f.write(f"Full Success:      {succeeded} ({succeeded / total * 100:.1f}%)\n")
            f.write(f"Partial Success:   {partial} ({partial / total * 100:.1f}%)\n")
            f.write(f"Failed:            {failed} ({failed / total * 100:.1f}%)\n")
            f.write(f"Errors:            {errored} ({errored / total * 100:.1f}%)\n")
            f.write(f"Infra Errors:      {infra_errored} ({infra_errored / total * 100:.1f}%)\n")
            f.write(f"\nAvg Duration:   {avg_dur:.2f} sec\n")
            f.write(f"Min Duration:   {min_dur:.2f} sec\n")
            f.write(f"Max Duration:   {max_dur:.2f} sec\n\n")

            f.write("BREAKDOWN BY TASK TYPE\n")
            f.write("-" * 40 + "\n")
            for task_type, counts in sorted(by_type.items()):
                subtotal = sum(counts.values())
                if subtotal == 0:
                    continue
                f.write(f"{task_type}:\n")
                for status, cnt in counts.items():
                    f.write(f"  {status:12s} {cnt} ({cnt / subtotal * 100:.1f}%)\n")

            # Failed jobs detail
            failed_jobs = [r for r in self.results if r.status not in ("success", "partial")]
            if failed_jobs:
                f.write("\nFAILED JOBS DETAIL\n")
                f.write("-" * 40 + "\n")
                for r in failed_jobs[:20]:
                    f.write(f"\nJob ID: {r.job_id}\n")
                    f.write(f"  Type: {r.task_type} | Region: {r.region}\n")
                    f.write(f"  Status: {r.status} | Error Type: {r.error_type or 'unknown'}\n")
                    ti = r.tile_info
                    f.write(f"  Tile: Year={ti.get('year')}, X={ti.get('X_tile')}, Y={ti.get('Y_tile')}\n")
                    f.write(f"  Error: {r.error_message[:200] if r.error_message else 'Unknown'}\n")
                if len(failed_jobs) > 20:
                    f.write(f"\n... and {len(failed_jobs) - 20} more failed jobs\n")

    def to_tile_results(self, step_name: str) -> dict[str, "StepResult"]:
        """Convert tracked job results into per-tile :class:`StepResult` objects.

        Groups results by tile key (``{X_tile}X{Y_tile}Y``).  If all jobs for
        a tile succeeded the step is ``"success"``; if any failed it's
        ``"failed"``.
        """
        from gri_tile_pipeline.tracking.run_metadata import StepResult

        tile_map: Dict[str, StepResult] = {}
        for r in self.results:
            if r.task_type.upper() not in _task_types_for_step(step_name):
                continue
            key = f"{r.tile_info.get('X_tile')}X{r.tile_info.get('Y_tile')}Y"
            existing = tile_map.get(key)
            if existing is None:
                tile_map[key] = StepResult(
                    status="success" if r.status in ("success", "partial") else "failed",
                    duration_sec=r.duration_sec,
                    error=r.error_message,
                )
            else:
                # Merge: any failure taints the tile result
                if r.status not in ("success", "partial"):
                    existing.status = "failed"
                    existing.error = existing.error or r.error_message
                if r.duration_sec and existing.duration_sec:
                    existing.duration_sec += r.duration_sec

        return tile_map

    def print_summary(self) -> None:
        """Print a quick summary to console."""
        total = len(self.results)
        if total == 0:
            logger.info("No jobs were executed.")
            return

        succeeded = sum(1 for r in self.results if r.status == "success")
        partial = sum(1 for r in self.results if r.status == "partial")
        failed = total - succeeded - partial

        logger.info(
            f"Job summary: {succeeded} succeeded, {partial} partial, {failed} failed "
            f"out of {total} total"
        )

        by_type: Dict[str, Dict[str, int]] = {}
        for r in self.results:
            key = f"{r.task_type}-{r.region}"
            bucket = by_type.setdefault(key, {"success": 0, "partial": 0, "failed": 0})
            if r.status == "success":
                bucket["success"] += 1
            elif r.status == "partial":
                bucket["partial"] += 1
            else:
                bucket["failed"] += 1

        for key, counts in sorted(by_type.items()):
            tot = sum(counts.values())
            logger.info(
                f"  {key}: {counts['success']} full + {counts['partial']} partial / {tot} total"
            )


def process_result(
    tracker: JobTracker,
    job_id: str,
    task_type: str,
    region: str,
    tile_info: Dict[str, Any],
    result: Any,
    stats: Optional[Dict[str, Any]],
    duration: Optional[float],
) -> None:
    """Process a single Lithops result and add to tracker."""
    if isinstance(result, dict):
        status = result.get("status", "success")
        tracker.add_result(
            JobResult(
                job_id=job_id,
                task_type=task_type,
                region=region,
                tile_info=tile_info,
                status=status,
                duration_sec=duration,
                result_data=result,
                stats=stats,
                quarters_succeeded=result.get("quarters_succeeded"),
                quarters_failed=result.get("quarters_failed"),
                quarter_details=result.get("quarter_details"),
                total_retry_count=result.get("total_retry_count"),
                stac_search_attempts=result.get("stac_search_attempts"),
                error_message=result.get("error_message"),
                error_type=result.get("error_type"),
                error_traceback=result.get("error_traceback"),
                debug_info=result.get("debug_info"),
                orbit_selected=result.get("orbit_selected"),
                orbit_stats=result.get("orbit_stats"),
            )
        )
    elif isinstance(result, Exception):
        tracker.add_result(
            JobResult(
                job_id=job_id,
                task_type=task_type,
                region=region,
                tile_info=tile_info,
                status="error",
                duration_sec=duration,
                error_message=str(result),
                error_type="unknown",
                stats=stats,
            )
        )
    else:
        tracker.add_result(
            JobResult(
                job_id=job_id,
                task_type=task_type,
                region=region,
                tile_info=tile_info,
                status="success",
                duration_sec=duration,
                result_data={"raw_result": str(result)},
                stats=stats,
            )
        )


def wait_all_with_tracking(
    retry_exec,
    futures: List[Tuple[Any, str, str, Dict[str, Any]]],
    tracker: JobTracker,
) -> List[Any]:
    """Wait for all Lithops retrying futures and record results in *tracker*.

    Args:
        retry_exec: ``lithops.RetryingFunctionExecutor``
        futures: List of ``(RetryingFuture, task_type, region, tile_info)``
        tracker: JobTracker instance

    Returns:
        List of raw results (dict or None).
    """
    import traceback as tb

    wait_dur_sec = 10 if len(futures) > 1000 else 3
    future_objects = [f[0] for f in futures]

    try:
        done, pending = retry_exec.wait(
            future_objects, throw_except=False, wait_dur_sec=wait_dur_sec
        )
    except KeyError as e:
        if "exc_info" in str(e):
            logger.warning(f"Lithops status error — falling back to per-future: {e}")
            results: List[Any] = []
            for rf, task_type, region, tile_info in futures:
                job_id = f"{task_type}_{tile_info['X_tile']}_{tile_info['Y_tile']}_{tile_info['year']}"
                try:
                    result = rf.response_future.result(throw_except=False)
                    stats = getattr(rf.response_future, "stats", None)
                    process_result(tracker, job_id, task_type, region, tile_info, result, stats, None)
                    results.append(result)
                except Exception as ex:
                    tracker.add_result(
                        JobResult(
                            job_id=job_id,
                            task_type=task_type,
                            region=region,
                            tile_info=tile_info,
                            status="infra_error",
                            error_message=str(ex),
                            error_traceback=tb.format_exc(),
                        )
                    )
                    results.append({"status": "infra_error", "error_message": str(ex)})
            return results
        raise

    assert len(pending) == 0

    results = []
    future_map = {id(f[0]): (f[1], f[2], f[3]) for f in futures}

    for rf in done:
        task_type, region, tile_info = future_map[id(rf)]
        job_id = f"{task_type}_{tile_info['X_tile']}_{tile_info['Y_tile']}_{tile_info['year']}"
        try:
            result = rf.response_future.result(throw_except=False)
            stats = getattr(rf.response_future, "stats", None)
            duration = (stats or {}).get("worker_exec_time")
            process_result(tracker, job_id, task_type, region, tile_info, result, stats, duration)
            results.append(result)
        except Exception as e:
            tracker.add_result(
                JobResult(
                    job_id=job_id,
                    task_type=task_type,
                    region=region,
                    tile_info=tile_info,
                    status="infra_error",
                    error_message=str(e),
                    error_traceback=tb.format_exc(),
                )
            )
            results.append({"status": "infra_error", "error_message": str(e)})

    return results


def _task_types_for_step(step_name: str) -> set[str]:
    """Map a pipeline step name to expected JobResult task_type values."""
    mapping = {
        "download_ard": {"DEM", "S1", "S2"},
        "download_s1_rtc": {"S1_RTC"},
        "predict": {"PREDICT"},
    }
    return mapping.get(step_name, {step_name.upper()})


def get_per_tile_status(tracker: JobTracker) -> dict[str, str]:
    """Return ``{tile_key: "success"|"failed"}`` from tracker results."""
    status: Dict[str, str] = {}
    for r in tracker.results:
        key = f"{r.tile_info.get('X_tile')}X{r.tile_info.get('Y_tile')}Y"
        if key not in status:
            status[key] = "success"
        if r.status not in ("success", "partial"):
            status[key] = "failed"
    return status
