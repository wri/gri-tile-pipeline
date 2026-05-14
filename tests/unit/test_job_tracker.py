"""Unit tests for tracking/job_tracker.py."""

import json
import os

import pytest

from gri_tile_pipeline.tracking.job_result import JobResult
from gri_tile_pipeline.tracking.job_tracker import (
    JobTracker,
    get_per_tile_status,
    wait_all_with_tracking,
)


def _make_result(status: str, x: int = 1000, y: int = 871) -> JobResult:
    return JobResult(
        job_id=f"test_{x}_{y}",
        task_type="DEM",
        region="us-west-2",
        tile_info={"year": 2024, "X_tile": x, "Y_tile": y, "lon": -73.5, "lat": 45.5},
        status=status,
        duration_sec=10.5,
    )


class TestJobTracker:
    def test_add_result(self, tmp_path):
        tracker = JobTracker(output_dir=str(tmp_path))
        tracker.add_result(_make_result("success"))
        assert len(tracker.results) == 1

    def test_save_reports_creates_files(self, tmp_path):
        tracker = JobTracker(output_dir=str(tmp_path))
        tracker.add_result(_make_result("success"))
        tracker.add_result(_make_result("failed"))
        tracker.save_reports()

        files = os.listdir(str(tmp_path))
        json_files = [f for f in files if f.endswith(".json")]
        csv_files = [f for f in files if f.endswith(".csv")]
        txt_files = [f for f in files if f.endswith(".txt")]

        assert len(json_files) >= 1  # job_report + failed_jobs
        assert len(csv_files) == 1
        assert len(txt_files) == 1

    def test_save_reports_valid_json(self, tmp_path):
        tracker = JobTracker(output_dir=str(tmp_path))
        tracker.add_result(_make_result("success"))
        tracker.save_reports()

        json_files = [f for f in os.listdir(str(tmp_path)) if f.startswith("job_report") and f.endswith(".json")]
        assert len(json_files) == 1
        with open(tmp_path / json_files[0]) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_empty_tracker_reports(self, tmp_path):
        tracker = JobTracker(output_dir=str(tmp_path))
        tracker.save_reports()
        # Should not raise


class TestGetPerTileStatus:
    def test_success(self, tmp_path):
        tracker = JobTracker(output_dir=str(tmp_path))
        tracker.add_result(_make_result("success", 1000, 871))
        status = get_per_tile_status(tracker)
        assert status["1000X871Y"] == "success"

    def test_failed(self, tmp_path):
        tracker = JobTracker(output_dir=str(tmp_path))
        tracker.add_result(_make_result("success", 1000, 871))
        tracker.add_result(_make_result("failed", 1000, 871))
        status = get_per_tile_status(tracker)
        assert status["1000X871Y"] == "failed"

    def test_multiple_tiles(self, tmp_path):
        tracker = JobTracker(output_dir=str(tmp_path))
        tracker.add_result(_make_result("success", 1000, 871))
        tracker.add_result(_make_result("failed", 1001, 872))
        status = get_per_tile_status(tracker)
        assert status["1000X871Y"] == "success"
        assert status["1001X872Y"] == "failed"


class _FakeResponseFuture:
    """Stand-in for lithops.future.ResponseFuture in the per-future fallback path.

    The fallback in wait_all_with_tracking calls .result() and reads .stats —
    both behaviors are reproduced here.
    """

    def __init__(self, result_value=None, raises=None):
        self._result_value = result_value
        self._raises = raises
        self.stats = {"worker_exec_time": 1.23}

    def result(self, throw_except=False):
        if self._raises is not None:
            raise self._raises
        return self._result_value


class _FakeRetryingFuture:
    def __init__(self, rf):
        self.response_future = rf


class _FakeRetryExec:
    """retry_exec.wait() raises a Lithops-style KeyError on the first call.

    Mimics the partial-call_status symptom that prompted the fallback path.
    """

    def __init__(self, key):
        self._key = key

    def wait(self, *args, **kwargs):
        raise KeyError(self._key)


class TestWaitAllWithTrackingFallback:
    """Per-future fallback when retry_exec.wait raises a Lithops KeyError.

    See job_tracker.wait_all_with_tracking — the partial-call_status JSON
    in Lithops trips a `_call_status[key]` access; recognized keys fall
    back to per-future polling so one bad status doesn't take the run down.
    """

    @pytest.mark.parametrize("key", ["exc_info", "func_result_size"])
    def test_recognized_key_falls_back(self, tmp_path, key):
        tracker = JobTracker(output_dir=str(tmp_path))
        rf_ok = _FakeRetryingFuture(_FakeResponseFuture(result_value={"out": "ok"}))
        rf_bad = _FakeRetryingFuture(_FakeResponseFuture(raises=RuntimeError("boom")))
        futures = [
            (rf_ok,  "DEM", "us-west-2", {"X_tile": 1, "Y_tile": 2, "year": 2024, "lon": 0.0, "lat": 0.0}),
            (rf_bad, "DEM", "us-west-2", {"X_tile": 3, "Y_tile": 4, "year": 2024, "lon": 0.0, "lat": 0.0}),
        ]

        results = wait_all_with_tracking(_FakeRetryExec(key), futures, tracker)

        assert len(results) == 2
        statuses = {r.tile_info["X_tile"]: r.status for r in tracker.results}
        assert statuses[3] == "infra_error"
        # The successful future's status comes from process_result() based on
        # the result payload — we only assert the bad one tagged correctly,
        # since that's the behavior under test.

    def test_unknown_key_reraises(self, tmp_path):
        tracker = JobTracker(output_dir=str(tmp_path))
        rf = _FakeRetryingFuture(_FakeResponseFuture(result_value={"out": "ok"}))
        futures = [
            (rf, "DEM", "us-west-2", {"X_tile": 1, "Y_tile": 2, "year": 2024, "lon": 0.0, "lat": 0.0}),
        ]

        with pytest.raises(KeyError):
            wait_all_with_tracking(_FakeRetryExec("unrelated_bug_key"), futures, tracker)
