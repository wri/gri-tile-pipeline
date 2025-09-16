#!/usr/bin/env python3
"""
Fan out DEM + S1 (eu-central-1) and S2 (us-west-2) downloads with Lithops.
Enhanced with detailed job tracking and reporting.

Usage:
  python lithops_data_download_job.py tiles.csv --dest s3://my-bucket/prefix --plot
"""

from __future__ import annotations

import os
import csv
import yaml
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import traceback

import lithops
from lithops import FunctionExecutor
from lithops.retries import RetryingFuture

import loaders
from loaders import download_dem, download_s1, download_s2


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
    status: str  # 'success', 'failed', 'error'
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_sec: Optional[float] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None  # Lithops future.stats
    
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
                'duration_sec', 'error_message'
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
                    'error_message': r.error_message[:100] if r.error_message else None
                }
                writer.writerow(row)
    
    def _save_text_report(self, path: str):
        """Save a human-readable summary report"""
        total = len(self.results)
        succeeded = sum(1 for r in self.results if r.status == 'success')
        failed = sum(1 for r in self.results if r.status == 'failed')
        errored = sum(1 for r in self.results if r.status == 'error')
        
        # Group by task type
        by_type = {}
        for r in self.results:
            if r.task_type not in by_type:
                by_type[r.task_type] = {'success': 0, 'failed': 0, 'error': 0}
            by_type[r.task_type][r.status] += 1
        
        # Calculate timing stats
        durations = [r.duration_sec for r in self.results if r.duration_sec]
        avg_duration = sum(durations) / len(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        
        with open(path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"LITHOPS JOB EXECUTION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("OVERALL SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Jobs:     {total}\n")
            f.write(f"Succeeded:      {succeeded} ({succeeded/total*100:.1f}%)\n")
            f.write(f"Failed:         {failed} ({failed/total*100:.1f}%)\n")
            f.write(f"Errors:         {errored} ({errored/total*100:.1f}%)\n")
            f.write(f"\n")
            f.write(f"Avg Duration:   {avg_duration:.2f} sec\n")
            f.write(f"Min Duration:   {min_duration:.2f} sec\n")
            f.write(f"Max Duration:   {max_duration:.2f} sec\n")
            f.write("\n")
            
            f.write("BREAKDOWN BY TASK TYPE\n")
            f.write("-" * 40 + "\n")
            for task_type, counts in sorted(by_type.items()):
                subtotal = sum(counts.values())
                f.write(f"{task_type}:\n")
                f.write(f"  Total:     {subtotal}\n")
                f.write(f"  Success:   {counts['success']} ({counts['success']/subtotal*100:.1f}%)\n")
                f.write(f"  Failed:    {counts['failed']} ({counts['failed']/subtotal*100:.1f}%)\n")
                f.write(f"  Error:     {counts['error']} ({counts['error']/subtotal*100:.1f}%)\n")
            
            # List failed jobs
            failed_jobs = [r for r in self.results if r.status != 'success']
            if failed_jobs:
                f.write("\n")
                f.write("FAILED JOBS DETAIL\n")
                f.write("-" * 40 + "\n")
                for r in failed_jobs[:20]:  # Show first 20
                    f.write(f"\nJob ID: {r.job_id}\n")
                    f.write(f"  Type: {r.task_type} | Region: {r.region}\n")
                    f.write(f"  Tile: Year={r.tile_info.get('year')}, "
                           f"X={r.tile_info.get('X_tile')}, Y={r.tile_info.get('Y_tile')}\n")
                    f.write(f"  Error: {r.error_message[:200] if r.error_message else 'Unknown'}\n")
                
                if len(failed_jobs) > 20:
                    f.write(f"\n... and {len(failed_jobs) - 20} more failed jobs\n")
    
    def print_summary(self):
        """Print a quick summary to console"""
        total = len(self.results)
        succeeded = sum(1 for r in self.results if r.status == 'success')
        failed = total - succeeded
        
        print("\n" + "=" * 50)
        print("JOB EXECUTION SUMMARY")
        print("=" * 50)
        print(f"Total Jobs:    {total}")
        print(f"✅ Succeeded:  {succeeded} ({succeeded/total*100:.1f}%)")
        print(f"❌ Failed:     {failed} ({failed/total*100:.1f}%)")
        
        # Show breakdown by type
        by_type = {}
        for r in self.results:
            key = f"{r.task_type}-{r.region}"
            if key not in by_type:
                by_type[key] = {'success': 0, 'failed': 0}
            if r.status == 'success':
                by_type[key]['success'] += 1
            else:
                by_type[key]['failed'] += 1
        
        print("\nBy Task Type:")
        for key, counts in sorted(by_type.items()):
            total_type = counts['success'] + counts['failed']
            print(f"  {key}: {counts['success']}/{total_type} succeeded")


# ----------------------------
# Worker entrypoints (top-level)
# ----------------------------

def _run_dem(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    from loaders.download_dem import run
    return run(**kwargs)

def _run_s1(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    from loaders.download_s1 import run
    return run(**kwargs)

def _run_s2(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    from loaders.download_s2 import run
    return run(**kwargs)


# ----------------------------
# Helper utilities
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
                    except:
                        pass
                    
                    if isinstance(result, Exception):
                        tracker.add_result(JobResult(
                            job_id=job_id,
                            task_type=task_type,
                            region=region,
                            tile_info=tile_info,
                            status='failed',
                            error_message=str(result),
                            error_traceback=traceback.format_exc(),
                            stats=stats
                        ))
                        results.append(None)
                    else:
                        tracker.add_result(JobResult(
                            job_id=job_id,
                            task_type=task_type,
                            region=region,
                            tile_info=tile_info,
                            status='success',
                            result_data=result,
                            stats=stats
                        ))
                        results.append(result)
                except Exception as ex:
                    tracker.add_result(JobResult(
                        job_id=job_id,
                        task_type=task_type,
                        region=region,
                        tile_info=tile_info,
                        status='error',
                        error_message=str(ex),
                        error_traceback=traceback.format_exc()
                    ))
                    results.append(None)
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
            except:
                pass
            
            # Calculate duration if stats available
            duration = None
            if stats and 'worker_exec_time' in stats:
                duration = stats['worker_exec_time']
            
            if isinstance(result, Exception):
                tracker.add_result(JobResult(
                    job_id=job_id,
                    task_type=task_type,
                    region=region,
                    tile_info=tile_info,
                    status='failed',
                    duration_sec=duration,
                    error_message=str(result),
                    stats=stats
                ))
                results.append(None)
            else:
                tracker.add_result(JobResult(
                    job_id=job_id,
                    task_type=task_type,
                    region=region,
                    tile_info=tile_info,
                    status='success',
                    duration_sec=duration,
                    result_data=result if isinstance(result, dict) else {'result': result},
                    stats=stats
                ))
                results.append(result)
                
        except Exception as e:
            tracker.add_result(JobResult(
                job_id=job_id,
                task_type=task_type,
                region=region,
                tile_info=tile_info,
                status='error',
                error_message=str(e),
                error_traceback=traceback.format_exc()
            ))
            results.append(None)
    
    return results


# ----------------------------
# CLI / Orchestrator
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles_csv", help="CSV with columns: Year,X,Y,Y_tile,X_tile")
    ap.add_argument("--dest", help="Output root (e.g. s3://bucket/prefix or local dir)",
                    default=os.environ.get("DEST"))
    ap.add_argument("--runtime", default="ttc-loaders-dev",
                    help="Lithops Lambda runtime name/tag")
    ap.add_argument("--mem", type=int, default=4096,
                    help="Lambda memory in MB (runtime_memory)")
    ap.add_argument("--retries", type=int, default=3,
                    help="Number of automatic retries per task")
    ap.add_argument("--usw2-cfg", default=".lithops/config.usw2.yaml",
                    help="Lithops config YAML for us-west-2")
    ap.add_argument("--euc1-cfg", default=".lithops/config.euc1.yaml",
                    help="Lithops config YAML for eu-central-1")
    ap.add_argument("--plot", action="store_true",
                    help="Save execution stats plots per region")
    ap.add_argument("--plot-dir", default="plots",
                    help="Directory to save plots when --plot is used")
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
            "debug": bool(args.debug),
        })

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
    for kw in base_kwargs:
        tile_info = {k: kw[k] for k in ['year', 'lon', 'lat', 'X_tile', 'Y_tile']}
        
        futures_euc1.append((
            RetryingFuture(
                fexec_euc1.call_async(_run_dem, (kw,), include_modules=['loaders']),
                _run_dem, (kw,), retries=args.retries
            ),
            'DEM',
            'eu-central-1',
            tile_info
        ))
        futures_euc1.append((
            RetryingFuture(
                fexec_euc1.call_async(_run_s1, (kw,), include_modules=['loaders']),
                _run_s1, (kw,), retries=args.retries
            ),
            'S1',
            'eu-central-1',
            tile_info
        ))

    # Submit S2 to us-west-2 with tracking info
    futures_usw2: List[Tuple[RetryingFuture, str, str, Dict[str, Any]]] = []
    for kw in base_kwargs:
        tile_info = {k: kw[k] for k in ['year', 'lon', 'lat', 'X_tile', 'Y_tile']}
        
        futures_usw2.append((
            RetryingFuture(
                fexec_usw2.call_async(_run_s2, (kw,), include_modules=['loaders']),
                _run_s2, (kw,), retries=args.retries
            ),
            'S2',
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