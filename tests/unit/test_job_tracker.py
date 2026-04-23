"""Unit tests for tracking/job_tracker.py."""

import json
import os

import pytest

from gri_tile_pipeline.tracking.job_result import JobResult
from gri_tile_pipeline.tracking.job_tracker import JobTracker, get_per_tile_status


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
