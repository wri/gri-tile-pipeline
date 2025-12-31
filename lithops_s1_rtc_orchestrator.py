#!/usr/bin/env python3
"""
Fan out S1 RTC input data composites (VV/VH) with Lithops.
This script is modified to use the loaders/download_s1_rtc.py script to download S1 data from the Planetary Computer, using Lithops on aws to facilitate compute and storage.
Enhanced with detailed job tracking and reporting.

Usage:
  python lithops_s1_rtc_orchestrator.py tiles.csv --dest s3://my-bucket/prefix --plot
"""

from __future__ import annotations

import os
import csv
import yaml
import json
import argparse
import time
import random
import requests
from datetime import datetime, timezone

from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import traceback

import lithops
from lithops import FunctionExecutor
from lithops.retries import RetryingFuture

import loaders
from loaders import download_dem, download_s1, download_s2, download_s1_rtc


# ----------------------------
# Job tracking data structures
# ----------------------------

@dataclass
class JobResult:
    """Track individual job execution results"""
    job_id: str
    task_type: str  # 'DEM', 'S1', or 'S2'
    region: str  # 'eu-central-1' or 'us-west-2'
    tile_info: Dict[str, Any]  # year, lon, lat, X_tile, Y_tile
    status: str  # 'success', 'partial', 'failed', 'error', 'infra_error'
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_sec: Optional[float] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    error_type: Optional[str] = None  # 'retryable', 'permanent', None
    result_data: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None  # Lithops future.stats
    # New fields for partial success and retry tracking
    quarters_succeeded: Optional[List[str]] = None
    quarters_failed: Optional[List[str]] = None
    quarter_details: Optional[Dict[str, Any]] = None
    total_retry_count: Optional[int] = None
    stac_search_attempts: Optional[int] = None
    # Debug info for error investigation
    debug_info: Optional[Dict[str, Any]] = None
    orbit_selected: Optional[str] = None
    orbit_stats: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JobTracker:
    """Centralized job tracking and reporting"""
    
    def __init__(self, output_dir: str = "job_reports"):
        self.output_dir = output_dir
        self.results: List[JobResult] = []
        self.start_time = datetime.now()
        os.makedirs(output_dir, exist_ok=True)
        
    def add_result(self, result: JobResult):
        self.results.append(result)
    
    def save_reports(self):
        """Save multiple report formats"""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # 1. Detailed JSON report
        json_path = os.path.join(self.output_dir, f"job_report_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2, default=str)
        
        # 2. CSV summary
        csv_path = os.path.join(self.output_dir, f"job_summary_{timestamp}.csv")
        self._save_csv_summary(csv_path)
        
        # 3. Human-readable text report
        txt_path = os.path.join(self.output_dir, f"job_report_{timestamp}.txt")
        self._save_text_report(txt_path)
        
        # 4. Failed jobs only (if any)
        failed = [r for r in self.results if r.status != 'success']
        if failed:
            failed_path = os.path.join(self.output_dir, f"failed_jobs_{timestamp}.json")
            with open(failed_path, 'w') as f:
                json.dump([r.to_dict() for r in failed], f, indent=2, default=str)
        
        print(f"\n📊 Reports saved to {self.output_dir}/")
        print(f"  - Detailed JSON: {os.path.basename(json_path)}")
        print(f"  - CSV Summary: {os.path.basename(csv_path)}")
        print(f"  - Text Report: {os.path.basename(txt_path)}")
        if failed:
            print(f"  - Failed Jobs: {os.path.basename(failed_path)}")
    
    def _save_csv_summary(self, path: str):
        """Save a CSV with one row per job"""
        with open(path, 'w', newline='') as f:
            fieldnames = [
                'job_id', 'task_type', 'region', 'status',
                'year', 'lon', 'lat', 'X_tile', 'Y_tile',
                'duration_sec', 'error_type', 'total_retry_count',
                'quarters_succeeded', 'quarters_failed', 'error_message'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in self.results:
                row = {
                    'job_id': r.job_id,
                    'task_type': r.task_type,
                    'region': r.region,
                    'status': r.status,
                    'year': r.tile_info.get('year'),
                    'lon': r.tile_info.get('lon'),
                    'lat': r.tile_info.get('lat'),
                    'X_tile': r.tile_info.get('X_tile'),
                    'Y_tile': r.tile_info.get('Y_tile'),
                    'duration_sec': r.duration_sec,
                    'error_type': r.error_type,
                    'total_retry_count': r.total_retry_count,
                    'quarters_succeeded': ','.join(r.quarters_succeeded) if r.quarters_succeeded else None,
                    'quarters_failed': ','.join(r.quarters_failed) if r.quarters_failed else None,
                    'error_message': r.error_message[:100] if r.error_message else None
                }
                writer.writerow(row)
    
    def _save_text_report(self, path: str):
        """Save a human-readable summary report"""
        total = len(self.results)
        succeeded = sum(1 for r in self.results if r.status == 'success')
        partial = sum(1 for r in self.results if r.status == 'partial')
        failed = sum(1 for r in self.results if r.status == 'failed')
        errored = sum(1 for r in self.results if r.status == 'error')
        infra_errored = sum(1 for r in self.results if r.status == 'infra_error')

        # Error type breakdown
        retryable_failures = sum(
            1 for r in self.results
            if r.status in ('failed', 'error') and r.error_type == 'retryable'
        )
        permanent_failures = sum(
            1 for r in self.results
            if r.status in ('failed', 'error') and r.error_type == 'permanent'
        )

        # Aggregate retry statistics
        total_retries = sum(
            r.total_retry_count or 0
            for r in self.results
            if r.total_retry_count
        )

        # Group by task type
        by_type: Dict[str, Dict[str, int]] = {}
        for r in self.results:
            if r.task_type not in by_type:
                by_type[r.task_type] = {'success': 0, 'partial': 0, 'failed': 0, 'error': 0, 'infra_error': 0}
            if r.status in by_type[r.task_type]:
                by_type[r.task_type][r.status] += 1

        # Calculate timing stats
        durations = [r.duration_sec for r in self.results if r.duration_sec]
        avg_duration = sum(durations) / len(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        min_duration = min(durations) if durations else 0

        with open(path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("LITHOPS JOB EXECUTION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            f.write("OVERALL SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Jobs:        {total}\n")
            f.write(f"Full Success:      {succeeded} ({succeeded/total*100:.1f}%)\n")
            f.write(f"Partial Success:   {partial} ({partial/total*100:.1f}%)\n")
            f.write(f"Failed:            {failed} ({failed/total*100:.1f}%)\n")
            f.write(f"Errors:            {errored} ({errored/total*100:.1f}%)\n")
            f.write(f"Infra Errors:      {infra_errored} ({infra_errored/total*100:.1f}%)\n")
            f.write("\n")
            f.write(f"Retryable failures: {retryable_failures}\n")
            f.write(f"Permanent failures: {permanent_failures}\n")
            f.write(f"Total retries used: {total_retries}\n")
            f.write("\n")
            f.write(f"Avg Duration:   {avg_duration:.2f} sec\n")
            f.write(f"Min Duration:   {min_duration:.2f} sec\n")
            f.write(f"Max Duration:   {max_duration:.2f} sec\n")
            f.write("\n")

            f.write("BREAKDOWN BY TASK TYPE\n")
            f.write("-" * 40 + "\n")
            for task_type, counts in sorted(by_type.items()):
                subtotal = sum(counts.values())
                if subtotal == 0:
                    continue
                f.write(f"{task_type}:\n")
                f.write(f"  Total:     {subtotal}\n")
                f.write(f"  Success:   {counts['success']} ({counts['success']/subtotal*100:.1f}%)\n")
                f.write(f"  Partial:   {counts['partial']} ({counts['partial']/subtotal*100:.1f}%)\n")
                f.write(f"  Failed:    {counts['failed']} ({counts['failed']/subtotal*100:.1f}%)\n")
                f.write(f"  Error:     {counts['error']} ({counts['error']/subtotal*100:.1f}%)\n")

            # Partial success details
            partial_jobs = [r for r in self.results if r.status == 'partial']
            if partial_jobs:
                f.write("\n")
                f.write("PARTIAL SUCCESS DETAILS\n")
                f.write("-" * 40 + "\n")
                for r in partial_jobs[:10]:
                    f.write(f"\nJob ID: {r.job_id}\n")
                    f.write(f"  Succeeded quarters: {r.quarters_succeeded}\n")
                    f.write(f"  Failed quarters: {r.quarters_failed}\n")
                    if r.total_retry_count:
                        f.write(f"  Total retries: {r.total_retry_count}\n")
                if len(partial_jobs) > 10:
                    f.write(f"\n... and {len(partial_jobs) - 10} more partial jobs\n")

            # List failed jobs
            failed_jobs = [r for r in self.results if r.status not in ('success', 'partial')]
            if failed_jobs:
                f.write("\n")
                f.write("FAILED JOBS DETAIL\n")
                f.write("-" * 40 + "\n")
                for r in failed_jobs[:20]:  # Show first 20
                    f.write(f"\nJob ID: {r.job_id}\n")
                    f.write(f"  Type: {r.task_type} | Region: {r.region}\n")
                    f.write(f"  Status: {r.status} | Error Type: {r.error_type or 'unknown'}\n")
                    f.write(f"  Tile: Year={r.tile_info.get('year')}, "
                           f"X={r.tile_info.get('X_tile')}, Y={r.tile_info.get('Y_tile')}\n")
                    f.write(f"  Error: {r.error_message[:200] if r.error_message else 'Unknown'}\n")
                    # Include debug info for investigation
                    if r.debug_info:
                        f.write("  Debug Info:\n")
                        if r.debug_info.get('stac_url'):
                            f.write(f"    STAC URL: {r.debug_info['stac_url']}\n")
                        if r.debug_info.get('bbox'):
                            f.write(f"    Bbox: {r.debug_info['bbox']}\n")
                        if r.debug_info.get('items_found'):
                            f.write(f"    Items found: {r.debug_info['items_found']}\n")

                if len(failed_jobs) > 20:
                    f.write(f"\n... and {len(failed_jobs) - 20} more failed jobs\n")
    
    def print_summary(self):
        """Print a quick summary to console"""
        total = len(self.results)
        succeeded = sum(1 for r in self.results if r.status == 'success')
        partial = sum(1 for r in self.results if r.status == 'partial')
        failed = sum(1 for r in self.results if r.status in ('failed', 'error', 'infra_error'))

        # Error type counts
        retryable = sum(1 for r in self.results if r.error_type == 'retryable')
        permanent = sum(1 for r in self.results if r.error_type == 'permanent')

        # Total retries
        total_retries = sum(r.total_retry_count or 0 for r in self.results if r.total_retry_count)

        print("\n" + "=" * 50)
        print("JOB EXECUTION SUMMARY")
        print("=" * 50)
        print(f"Total Jobs:       {total}")
        print(f"Full Success:     {succeeded} ({succeeded/total*100:.1f}%)")
        print(f"Partial Success:  {partial} ({partial/total*100:.1f}%)")
        print(f"Failed:           {failed} ({failed/total*100:.1f}%)")

        if retryable or permanent:
            print(f"\nRetryable failures: {retryable}")
            print(f"Permanent failures: {permanent}")

        if total_retries:
            print(f"Total retries used: {total_retries}")

        # Show breakdown by type
        by_type: Dict[str, Dict[str, int]] = {}
        for r in self.results:
            key = f"{r.task_type}-{r.region}"
            if key not in by_type:
                by_type[key] = {'success': 0, 'partial': 0, 'failed': 0}
            if r.status == 'success':
                by_type[key]['success'] += 1
            elif r.status == 'partial':
                by_type[key]['partial'] += 1
            else:
                by_type[key]['failed'] += 1

        print("\nBy Task Type:")
        for key, counts in sorted(by_type.items()):
            total_type = counts['success'] + counts['partial'] + counts['failed']
            print(f"  {key}: {counts['success']} full + {counts['partial']} partial / {total_type} total")


# ----------------------------
# Worker entrypoints (top-level)
# ----------------------------

def _run_dem(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    from loaders.download_dem import run
    return run(**kwargs)

def _run_s1(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker entrypoint. Always returns a dict, never raises.
    Lithops will always receive a valid serializable result.
    The run() function handles all exceptions internally.
    """
    # De-synchronize Planetary Computer/STAC load across a large Lambda fan-out
    time.sleep(random.uniform(0.0, 2.0))
    from loaders.download_s1_rtc import run
    # run() never raises - always returns structured dict with status field
    return run(**kwargs)

def _run_s2(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    from loaders.download_s2 import run
    return run(**kwargs)


# ----------------------------
# Helper utilities

def get_pc_collection_token(collection_id: str, api_key: str | None = None) -> dict:
    """Fetch a Planetary Computer SAS token for a collection.

    Returns a dict like {"token": "...", "expiry": datetime|None}.
    """
    url = f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection_id}"
    headers = {}
    if api_key:
        headers["Ocp-Apim-Subscription-Key"] = api_key

    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    expiry = None
    # PC typically includes an ISO timestamp in msft:expiry; keep optional.
    exp = data.get("msft:expiry") or data.get("expiry") or data.get("expires")
    if isinstance(exp, str):
        try:
            expiry = datetime.fromisoformat(exp.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            expiry = None
    return {"token": data["token"], "expiry": expiry}


def _load_token_cache(cache_path: str) -> dict | None:
    """Load a cached SAS token blob from disk.

    Expected schema:
      {
        "collection_id": "sentinel-1-rtc",
        "token": "sv=...",
        "expiry": "2025-12-30T12:34:56+00:00"
      }
    """
    try:
        if not cache_path or not os.path.exists(cache_path):
            return None
        with open(cache_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _save_token_cache(cache_path: str, collection_id: str, token: str, expiry: datetime | None) -> None:
    """Persist SAS token + expiry for reuse between Lithops runs."""
    if not cache_path:
        return
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        payload = {
            "collection_id": collection_id,
            "token": token,
            "expiry": expiry.isoformat() if isinstance(expiry, datetime) else None,
        }
        with open(cache_path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"[pc-token] Warning: failed to write cache to {cache_path}: {e}")


def _parse_cached_expiry(expiry_val: Any) -> datetime | None:
    if isinstance(expiry_val, datetime):
        return expiry_val.astimezone(timezone.utc)
    if isinstance(expiry_val, str) and expiry_val.strip():
        try:
            return datetime.fromisoformat(expiry_val.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return None
    return None


def ensure_pc_collection_token(
    collection_id: str,
    cache_path: str,
    api_key: str | None = None,
    min_ttl_seconds: int = 20 * 60,
) -> dict:
    """Return a valid SAS token, reusing a cached one if it has sufficient TTL.

    If no cached token exists, or it is expired / too close to expiry, fetch a new one and cache it.
    """
    now = datetime.now(timezone.utc)

    cached = _load_token_cache(cache_path)
    if cached and cached.get("collection_id") == collection_id and isinstance(cached.get("token"), str):
        expiry = _parse_cached_expiry(cached.get("expiry"))
        if expiry:
            remaining = (expiry - now).total_seconds()
            if remaining > min_ttl_seconds:
                return {"token": cached["token"], "expiry": expiry, "source": "cache"}
        # If expiry missing or too close, treat as invalid and refresh.

    token_info = get_pc_collection_token(collection_id, api_key=api_key)
    _save_token_cache(cache_path, collection_id, token_info["token"], token_info.get("expiry"))
    token_info["source"] = "fresh"
    return token_info


def _format_ttl(expiry: datetime | None) -> str:
    if not isinstance(expiry, datetime):
        return "unknown remaining time"
    now = datetime.now(timezone.utc)
    remaining = max(0, int((expiry - now).total_seconds()))
    hrs = remaining // 3600
    mins = (remaining % 3600) // 60
    secs = remaining % 60
    if hrs > 0:
        return f"{hrs}h {mins}m"
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"



# ----------------------------

def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _read_tiles_csv(path: str) -> List[Dict[str, Any]]:
    """
    Expect columns: Year, X, Y, Y_tile, X_tile
      - X = lon (float)
      - Y = lat (float)
    """
    rows: List[Dict[str, Any]] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        required = {"Year", "X", "Y", "Y_tile", "X_tile"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")

        for row in r:
            rows.append({
                "year": int(row["Year"]),
                "lon": float(row["X"]),
                "lat": float(row["Y"]),
                "X_tile": int(row["X_tile"]),
                "Y_tile": int(row["Y_tile"]),
            })
    return rows

def _wait_all_with_tracking(
    retry_exec: lithops.RetryingFunctionExecutor,
    futures: List[Tuple[RetryingFuture, str, str, Dict[str, Any]]],
    tracker: JobTracker
) -> List[Any]:
    """
    Enhanced wait function with detailed tracking.

    Now checks result['status'] field instead of isinstance(Exception) since
    workers never crash - they always return structured dicts.

    Args:
        retry_exec: The retrying executor
        futures: List of tuples (future, task_type, region, tile_info)
        tracker: JobTracker instance

    Returns:
        List of results
    """
    wait_dur_sec = 10 if len(futures) > 1000 else 3

    # Extract just the futures for wait()
    future_objects = [f[0] for f in futures]

    try:
        done, pending = retry_exec.wait(
            future_objects, throw_except=False, wait_dur_sec=wait_dur_sec
        )
    except KeyError as e:
        if 'exc_info' in str(e):
            print(f"Warning: Lithops status error - {e}")
            # Fall back to getting results individually
            results = []
            for rf, task_type, region, tile_info in futures:
                job_id = f"{task_type}_{tile_info['X_tile']}_{tile_info['Y_tile']}_{tile_info['year']}"
                try:
                    result = rf.response_future.result(throw_except=False)

                    # Try to get stats if available
                    stats = None
                    try:
                        stats = rf.response_future.stats
                    except Exception:
                        pass

                    _process_result(tracker, job_id, task_type, region, tile_info, result, stats, None)
                    results.append(result)
                except Exception as ex:
                    tracker.add_result(JobResult(
                        job_id=job_id,
                        task_type=task_type,
                        region=region,
                        tile_info=tile_info,
                        status='infra_error',
                        error_message=str(ex),
                        error_traceback=traceback.format_exc()
                    ))
                    results.append({"status": "infra_error", "error_message": str(ex)})
            return results
        else:
            raise

    assert len(pending) == 0

    # Process results with tracking
    results = []
    future_map = {id(f[0]): (f[1], f[2], f[3]) for f in futures}

    for rf in done:
        task_type, region, tile_info = future_map[id(rf)]
        job_id = f"{task_type}_{tile_info['X_tile']}_{tile_info['Y_tile']}_{tile_info['year']}"

        try:
            result = rf.response_future.result(throw_except=False)

            # Get execution stats
            stats = None
            try:
                stats = rf.response_future.stats
            except Exception:
                pass

            # Calculate duration if stats available
            duration = None
            if stats and 'worker_exec_time' in stats:
                duration = stats['worker_exec_time']

            _process_result(tracker, job_id, task_type, region, tile_info, result, stats, duration)
            results.append(result)

        except Exception as e:
            # Lithops infrastructure error (not worker error)
            tracker.add_result(JobResult(
                job_id=job_id,
                task_type=task_type,
                region=region,
                tile_info=tile_info,
                status='infra_error',
                error_message=str(e),
                error_traceback=traceback.format_exc()
            ))
            results.append({"status": "infra_error", "error_message": str(e)})

    return results


def _process_result(
    tracker: JobTracker,
    job_id: str,
    task_type: str,
    region: str,
    tile_info: Dict[str, Any],
    result: Any,
    stats: Optional[Dict[str, Any]],
    duration: Optional[float]
) -> None:
    """Process a single result and add to tracker."""
    # Result is always a dict now (worker never crashes)
    if isinstance(result, dict):
        status = result.get('status', 'success')

        tracker.add_result(JobResult(
            job_id=job_id,
            task_type=task_type,
            region=region,
            tile_info=tile_info,
            status=status,
            duration_sec=duration,
            result_data=result,
            stats=stats,
            quarters_succeeded=result.get('quarters_succeeded'),
            quarters_failed=result.get('quarters_failed'),
            quarter_details=result.get('quarter_details'),
            total_retry_count=result.get('total_retry_count'),
            stac_search_attempts=result.get('stac_search_attempts'),
            error_message=result.get('error_message'),
            error_type=result.get('error_type'),
            error_traceback=result.get('error_traceback'),
            debug_info=result.get('debug_info'),
            orbit_selected=result.get('orbit_selected'),
            orbit_stats=result.get('orbit_stats'),
        ))

    elif isinstance(result, Exception):
        # Should rarely happen now, but handle just in case
        tracker.add_result(JobResult(
            job_id=job_id,
            task_type=task_type,
            region=region,
            tile_info=tile_info,
            status='error',
            duration_sec=duration,
            error_message=str(result),
            error_type='unknown',
            stats=stats,
        ))

    else:
        # Unexpected result type
        tracker.add_result(JobResult(
            job_id=job_id,
            task_type=task_type,
            region=region,
            tile_info=tile_info,
            status='success',
            duration_sec=duration,
            result_data={'raw_result': str(result)},
            stats=stats,
        ))


# ----------------------------
# CLI / Orchestrator
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles_csv", help="CSV with columns: Year,X,Y,Y_tile,X_tile")
    ap.add_argument("--dest", help="Output root (e.g. s3://bucket/prefix or local dir)",
                    default=os.environ.get("DEST"))
    ap.add_argument("--runtime", default="ttc-s1-dev",
                    help="Lithops Lambda runtime name/tag")
    ap.add_argument("--mem", type=int, default=2048,
                    help="Lambda memory in MB (runtime_memory)")
    ap.add_argument("--retries", type=int, default=3,
                    help="Number of automatic retries per task")
    ap.add_argument("--usw2-cfg", default=".lithops/config.s1_usw2.yaml",
                    help="Lithops config YAML for us-west-2")
    ap.add_argument("--euc1-cfg", default=".lithops/config.euc1.yaml",
                    help="Lithops config YAML for eu-central-1")
    ap.add_argument("--plot", action="store_true",
                    help="Save execution stats plots per region")
    ap.add_argument("--plot-dir", default="plots",
                    help="Directory to save plots when --plot is used")
    ap.add_argument("--pc-token-cache", default=os.environ.get("PC_TOKEN_CACHE", ".pc_sas_token_cache.json"),
                    help="Path to cache the Planetary Computer SAS token (runner-side) between runs")
    ap.add_argument("--pc-token-min-ttl-minutes", type=int, default=int(os.environ.get("PC_TOKEN_MIN_TTL_MINUTES", "20")),
                    help="If cached token has less than this TTL remaining, refresh it before fan-out")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the interactive approval prompt and immediately fan out jobs")
    ap.add_argument("--report-dir", default="job_reports",
                    help="Directory to save job tracking reports")
    ap.add_argument("--debug", action="store_true",
                    help="Pass debug=True to the loaders' run()")
    ap.add_argument("--overwrite", action="store_true",
                    help="Accepted for parity; loaders may decide how to handle overwrites")
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    if not args.dest:
        raise SystemExit("Please pass --dest or set DEST environment variable")

    # Initialize job tracker
    tracker = JobTracker(args.report_dir)

    # Load per-region Lithops configs
    cfg_usw2 = _load_cfg(args.usw2_cfg)
    cfg_euc1 = _load_cfg(args.euc1_cfg)

    # Ensure runtime tag and memory are set
    cfg_usw2.setdefault("aws_lambda", {})["runtime"] = args.runtime
    cfg_euc1.setdefault("aws_lambda", {})["runtime"] = args.runtime

    tiles = _read_tiles_csv(args.tiles_csv)

    # Planetary Computer auth strategy:
    # Fetch a single SAS token for the Sentinel-1 RTC collection once in the local runner,
    # then pass it into every Lambda task to avoid per-item signing (and avoid 429 throttling).
    pc_api_key = os.getenv("PC_SDK_SUBSCRIPTION_KEY") or os.getenv("PC_SUBSCRIPTION_KEY")
    # Token lifecycle: reuse cached SAS token if still valid; otherwise fetch a fresh one.
    token_info = ensure_pc_collection_token(
        "sentinel-1-rtc",
        cache_path=args.pc_token_cache,
        api_key=pc_api_key,
        min_ttl_seconds=int(args.pc_token_min_ttl_minutes) * 60,
    )
    pc_sas_token = token_info["token"]
    pc_sas_expiry = token_info.get("expiry")
    token_src = token_info.get("source", "unknown")
    if isinstance(pc_sas_expiry, datetime):
        print(f"[pc-token] Using {token_src} SAS token for sentinel-1-rtc; expires at {pc_sas_expiry.isoformat()} UTC ({_format_ttl(pc_sas_expiry)} remaining).")
    else:
        print(f"[pc-token] Using {token_src} SAS token for sentinel-1-rtc; expiry unknown.")


    # Build per-task kwargs
    base_kwargs: List[Dict[str, Any]] = []
    for t in tiles:
        base_kwargs.append({
            "year": t["year"],
            "lon": t["lon"],
            "lat": t["lat"],
            "X_tile": t["X_tile"],
            "Y_tile": t["Y_tile"],
            "dest": args.dest,
            "sas_token": pc_sas_token,
            "debug": bool(args.debug),
        })


    # ---- Approval gate (pre-fan-out) ----
    planned_euc1 = 0  # hmm
    planned_usw2 = len(base_kwargs)                # S1 jobs (us-west-2)
    print(f"\nPlanned fan-out: {planned_euc1 + planned_usw2} jobs total")
    print(f"  EU-Central-1: {planned_euc1} jobs")
    print(f"  US-West-2:   {planned_usw2} jobs")
    if not args.yes:
        resp = input("Type 'yes' to proceed with Lambda fan-out (anything else will abort): ").strip().lower()
        if resp != "yes":
            raise SystemExit("Aborted before fan-out.")

    # Executors
    fexec_euc1 = FunctionExecutor(
        config=cfg_euc1, runtime=args.runtime, runtime_memory=args.mem
    )
    fexec_usw2 = FunctionExecutor(
        config=cfg_usw2, runtime=args.runtime, runtime_memory=args.mem
    )

    retry_euc1 = lithops.RetryingFunctionExecutor(fexec_euc1)
    retry_usw2 = lithops.RetryingFunctionExecutor(fexec_usw2)

    # Submit DEM + S1 to eu-central-1 with tracking info
    futures_euc1: List[Tuple[RetryingFuture, str, str, Dict[str, Any]]] = []


    # Submit S2 to us-west-2 with tracking info
    futures_usw2: List[Tuple[RetryingFuture, str, str, Dict[str, Any]]] = []
    for kw in base_kwargs:
        tile_info = {k: kw[k] for k in ['year', 'lon', 'lat', 'X_tile', 'Y_tile']}
        futures_usw2.append((
            RetryingFuture(
                fexec_usw2.call_async(_run_s1, (kw,), include_modules=['loaders']),
                _run_s1, (kw,), retries=args.retries
            ),
            'S1',
            'us-west-2',
            tile_info
        ))
    # Wait and collect with tracking
    print(f"\n🚀 Submitted {len(futures_euc1) + len(futures_usw2)} jobs total")
    print(f"   EU-Central-1: {len(futures_euc1)} jobs")
    print(f"   US-West-2: {len(futures_usw2)} jobs")
    
    results_euc1 = _wait_all_with_tracking(retry_euc1, futures_euc1, tracker) if futures_euc1 else []
    results_usw2 = _wait_all_with_tracking(retry_usw2, futures_usw2, tracker) if futures_usw2 else []
    
    # Optional plotting
    if args.plot:
        os.makedirs(args.plot_dir, exist_ok=True)
        try:
            fexec_euc1.plot(dst=os.path.join(args.plot_dir, "eu-central-1.png"))
        except Exception as e:
            print(f"[plot] eu-central-1 plot failed: {e}")
        try:
            fexec_usw2.plot(dst=os.path.join(args.plot_dir, "us-west-2.png"))
        except Exception as e:
            print(f"[plot] us-west-2 plot failed: {e}")

    # Save reports and print summary
    tracker.print_summary()
    tracker.save_reports()

if __name__ == "__main__":
    main()