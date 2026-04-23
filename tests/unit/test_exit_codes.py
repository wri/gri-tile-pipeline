"""Unit tests for exit_codes.py."""

import pytest

from gri_tile_pipeline.exit_codes import ExitCode, exit_code_from_tracker
from gri_tile_pipeline.tracking.job_result import JobResult
from gri_tile_pipeline.tracking.job_tracker import JobTracker


def _make_result(status: str) -> JobResult:
    return JobResult(
        job_id="test",
        task_type="DEM",
        region="us-west-2",
        tile_info={"year": 2024, "X_tile": 1000, "Y_tile": 871},
        status=status,
    )


class TestExitCode:
    def test_success_value(self):
        assert ExitCode.SUCCESS == 0

    def test_partial_failure_value(self):
        assert ExitCode.PARTIAL_FAILURE == 1

    def test_total_failure_value(self):
        assert ExitCode.TOTAL_FAILURE == 2


class TestExitCodeFromTracker:
    def test_all_success(self, tmp_path):
        tracker = JobTracker(output_dir=str(tmp_path))
        tracker.add_result(_make_result("success"))
        tracker.add_result(_make_result("success"))
        assert exit_code_from_tracker(tracker) == ExitCode.SUCCESS

    def test_partial_counts_as_success(self, tmp_path):
        tracker = JobTracker(output_dir=str(tmp_path))
        tracker.add_result(_make_result("success"))
        tracker.add_result(_make_result("partial"))
        assert exit_code_from_tracker(tracker) == ExitCode.SUCCESS

    def test_some_failed(self, tmp_path):
        tracker = JobTracker(output_dir=str(tmp_path))
        tracker.add_result(_make_result("success"))
        tracker.add_result(_make_result("failed"))
        assert exit_code_from_tracker(tracker) == ExitCode.PARTIAL_FAILURE

    def test_all_failed(self, tmp_path):
        tracker = JobTracker(output_dir=str(tmp_path))
        tracker.add_result(_make_result("failed"))
        tracker.add_result(_make_result("error"))
        assert exit_code_from_tracker(tracker) == ExitCode.TOTAL_FAILURE

    def test_empty_tracker(self, tmp_path):
        tracker = JobTracker(output_dir=str(tmp_path))
        assert exit_code_from_tracker(tracker) == ExitCode.SUCCESS
