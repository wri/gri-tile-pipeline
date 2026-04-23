"""Tests for PhaseTimer and the JobTracker phase-stats roll-up."""

import csv
import json
import os
import time

from gri_tile_pipeline.phase_timer import PhaseTimer, timed
from gri_tile_pipeline.tracking import JobTracker
from gri_tile_pipeline.tracking.job_result import JobResult


def test_phase_timer_accumulates_and_snapshots():
    t = PhaseTimer()
    with t.phase("a"):
        time.sleep(0.002)
    with t.phase("a"):
        time.sleep(0.002)
    with t.phase("b"):
        time.sleep(0.001)

    snap = t.as_dict()
    assert set(snap) == {"a", "b"}
    # "a" should have roughly 2x "b"
    assert snap["a"] > snap["b"] > 0


def test_timed_is_noop_when_timer_is_none():
    with timed(None, "unused"):
        pass  # does not raise, does not allocate a timer


def test_timed_delegates_when_timer_is_provided():
    t = PhaseTimer()
    with timed(t, "work"):
        time.sleep(0.001)
    assert "work" in t.as_dict()


def test_job_tracker_writes_phase_stats_csv(tmp_path):
    tracker = JobTracker(output_dir=str(tmp_path / "reports"), run_id="r1")
    tracker.add_result(
        JobResult(
            job_id="PREDICT_100_200_2024",
            task_type="PREDICT",
            region="us-east-1",
            tile_info={"year": 2024, "X_tile": 100, "Y_tile": 200},
            status="success",
            duration_sec=12.5,
            result_data={
                "phase_timings": {
                    "s3_download_hkl": 4.2,
                    "tf_inference": 3.8,
                    "cog_write": 0.6,
                },
                "wallclock_sec": 12.5,
            },
        )
    )
    tracker.add_result(
        JobResult(
            job_id="PREDICT_101_200_2024",
            task_type="PREDICT",
            region="us-east-1",
            tile_info={"year": 2024, "X_tile": 101, "Y_tile": 200},
            status="success",
            duration_sec=15.1,
            result_data={
                "phase_timings": {
                    "s3_download_hkl": 5.1,
                    "tf_inference": 4.9,
                    "cog_write": 0.7,
                },
                "wallclock_sec": 15.1,
            },
        )
    )

    run_dir = tracker.save_run_report(str(tmp_path / "runs"))
    phase_csv = os.path.join(run_dir, "phase_stats.csv")
    assert os.path.isfile(phase_csv)

    with open(phase_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["X_tile"] == "100"
    assert float(rows[0]["s3_download_hkl"]) == 4.2
    assert float(rows[0]["tf_inference"]) == 3.8

    with open(os.path.join(run_dir, "summary.json")) as f:
        summary = json.load(f)
    agg = summary["phase_percentiles_sec"]
    assert set(agg) == {"s3_download_hkl", "tf_inference", "cog_write"}
    assert agg["s3_download_hkl"]["n"] == 2
    assert agg["s3_download_hkl"]["mean"] > 0


def test_job_tracker_skips_phase_artifacts_when_no_timings(tmp_path):
    """Tiles without phase_timings do not produce a phase_stats.csv."""
    tracker = JobTracker(output_dir=str(tmp_path / "reports"), run_id="r2")
    tracker.add_result(
        JobResult(
            job_id="PREDICT_1_2_2024",
            task_type="PREDICT",
            region="us-east-1",
            tile_info={"year": 2024, "X_tile": 1, "Y_tile": 2},
            status="success",
            duration_sec=10.0,
            result_data={"status": "success"},
        )
    )
    run_dir = tracker.save_run_report(str(tmp_path / "runs"))
    assert not os.path.isfile(os.path.join(run_dir, "phase_stats.csv"))
    with open(os.path.join(run_dir, "summary.json")) as f:
        summary = json.load(f)
    assert "phase_percentiles_sec" not in summary
