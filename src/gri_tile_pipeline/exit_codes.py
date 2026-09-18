"""Structured exit codes for pipeline commands."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gri_tile_pipeline.tracking import JobTracker


class ExitCode(IntEnum):
    SUCCESS: int = 0
    PARTIAL_FAILURE: int = 1
    TOTAL_FAILURE: int = 2
    BAD_INPUT: int = 3
    MISSING_DEPENDENCY: int = 4
    USER_ABORT: int = 5
    NO_WORK: int = 6  # Nothing to process (all tiles exist)
    TILES_MISSING: int = 10  # check command: some tiles not yet on S3


def exit_code_from_tracker(tracker: JobTracker) -> ExitCode:
    """Derive an exit code from a :class:`JobTracker`'s results."""
    failed = sum(1 for r in tracker.results if r.status not in ("success", "partial"))
    if failed == len(tracker.results) and tracker.results:
        return ExitCode.TOTAL_FAILURE
    elif failed > 0:
        return ExitCode.PARTIAL_FAILURE
    return ExitCode.SUCCESS
