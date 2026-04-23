"""Tests for tracking.run_index."""

import json
from pathlib import Path

from gri_tile_pipeline.tracking.run_index import RunSummary, get_run, list_runs


def _write_summary(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(payload))


def test_list_runs_empty(tmp_path):
    assert list_runs(tmp_path) == []


def test_list_runs_missing_dir(tmp_path):
    assert list_runs(tmp_path / "does-not-exist") == []


def test_list_runs_sorted_newest_first(tmp_path):
    _write_summary(tmp_path / "older", {
        "run_id": "older", "step": "predict",
        "start_time": "2024-01-01T09:00:00",
        "n_jobs": 10, "n_success": 8, "n_partial": 0, "n_failed": 2,
    })
    _write_summary(tmp_path / "newer", {
        "run_id": "newer", "step": "download",
        "start_time": "2024-02-01T09:00:00",
        "n_jobs": 5, "n_success": 5, "n_partial": 0, "n_failed": 0,
    })
    summaries = list_runs(tmp_path)
    assert [s.run_id for s in summaries] == ["newer", "older"]


def test_list_runs_ignores_malformed(tmp_path):
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "summary.json").write_text("not json")
    _write_summary(tmp_path / "good", {
        "run_id": "good", "start_time": "2024-01-01",
        "n_jobs": 1, "n_success": 1,
    })
    summaries = list_runs(tmp_path)
    assert [s.run_id for s in summaries] == ["good"]


def test_get_run_returns_summary(tmp_path):
    _write_summary(tmp_path / "abc", {
        "run_id": "abc", "step": "predict",
        "n_jobs": 3, "n_success": 2, "n_partial": 0, "n_failed": 1,
        "by_task_type": {"PREDICT": {"success": 2, "failed": 1}},
    })
    s = get_run(tmp_path, "abc")
    assert isinstance(s, RunSummary)
    assert s.step == "predict"
    assert s.n_failed == 1
    assert s.by_task_type == {"PREDICT": {"success": 2, "failed": 1}}


def test_get_run_missing(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        get_run(tmp_path, "ghost")


def test_run_summary_paths(tmp_path):
    _write_summary(tmp_path / "x", {"run_id": "x", "n_jobs": 0})
    s = get_run(tmp_path, "x")
    assert s.failed_csv == tmp_path / "x" / "failed.csv"
    assert s.jobs_csv == tmp_path / "x" / "jobs.csv"
    assert s.report_md == tmp_path / "x" / "report.md"


def test_save_run_report_writes_canonical_files(tmp_path):
    """Integration test: JobTracker.save_run_report produces all four canonical files."""
    from gri_tile_pipeline.tracking.job_result import JobResult
    from gri_tile_pipeline.tracking.job_tracker import JobTracker

    tracker = JobTracker(
        output_dir=str(tmp_path / "job_reports"),
        run_id="testrun42",
        step_name="predict",
    )
    tracker.add_result(JobResult(
        job_id="PREDICT_1_2_2023", task_type="PREDICT", region="us-west-2",
        tile_info={"year": 2023, "lon": 0.1, "lat": 0.2, "X_tile": 1, "Y_tile": 2},
        status="success", duration_sec=100.0,
    ))
    tracker.add_result(JobResult(
        job_id="PREDICT_3_4_2023", task_type="PREDICT", region="us-west-2",
        tile_info={"year": 2023, "lon": 0.3, "lat": 0.4, "X_tile": 3, "Y_tile": 4},
        status="failed", duration_sec=50.0, error_message="boom",
    ))

    run_dir_str = tracker.save_run_report(str(tmp_path / "runs"))
    run_dir = Path(run_dir_str)

    assert run_dir.name == "testrun42"
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "jobs.csv").is_file()
    assert (run_dir / "failed.csv").is_file()
    assert (run_dir / "report.md").is_file()

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["run_id"] == "testrun42"
    assert summary["step"] == "predict"
    assert summary["n_jobs"] == 2
    assert summary["n_success"] == 1
    assert summary["n_failed"] == 1
    assert summary["n_failed_tiles"] == 1
