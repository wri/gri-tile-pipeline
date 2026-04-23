"""Discover and summarize past runs stashed under a run-history directory.

Each run lives at ``<run_history_dir>/<run_id>/`` and contains at least a
``summary.json`` produced by :meth:`JobTracker.save_run_report`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunSummary:
    run_id: str
    path: Path
    step: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    duration_sec: float | None = None
    n_jobs: int = 0
    n_success: int = 0
    n_partial: int = 0
    n_failed: int = 0
    n_failed_tiles: int = 0
    by_task_type: dict[str, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def from_dir(cls, run_dir: Path) -> "RunSummary | None":
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            return None
        try:
            data: dict[str, Any] = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        return cls(
            run_id=data.get("run_id") or run_dir.name,
            path=run_dir,
            step=data.get("step"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            duration_sec=data.get("duration_sec"),
            n_jobs=int(data.get("n_jobs", 0)),
            n_success=int(data.get("n_success", 0)),
            n_partial=int(data.get("n_partial", 0)),
            n_failed=int(data.get("n_failed", 0)),
            n_failed_tiles=int(data.get("n_failed_tiles", 0)),
            by_task_type=data.get("by_task_type") or {},
        )

    @property
    def failed_csv(self) -> Path:
        return self.path / "failed.csv"

    @property
    def jobs_csv(self) -> Path:
        return self.path / "jobs.csv"

    @property
    def report_md(self) -> Path:
        return self.path / "report.md"

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": str(self.path),
            "step": self.step,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_sec": self.duration_sec,
            "n_jobs": self.n_jobs,
            "n_success": self.n_success,
            "n_partial": self.n_partial,
            "n_failed": self.n_failed,
            "n_failed_tiles": self.n_failed_tiles,
            "by_task_type": self.by_task_type,
        }


def list_runs(run_history_dir: str | Path) -> list[RunSummary]:
    """Return all runs under *run_history_dir*, newest first."""
    root = Path(run_history_dir)
    if not root.is_dir():
        return []
    summaries: list[RunSummary] = []
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        summary = RunSummary.from_dir(sub)
        if summary is not None:
            summaries.append(summary)
    summaries.sort(key=lambda r: r.start_time or "", reverse=True)
    return summaries


def get_run(run_history_dir: str | Path, run_id: str) -> RunSummary:
    """Load a specific run's summary. Raises FileNotFoundError if missing."""
    run_dir = Path(run_history_dir) / run_id
    summary = RunSummary.from_dir(run_dir)
    if summary is None:
        raise FileNotFoundError(f"No run {run_id!r} under {run_history_dir!r}")
    return summary
