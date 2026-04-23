"""Unified JobResult dataclass — superset of fields from both orchestrators."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class JobResult:
    """Track individual job execution results.

    This is the unified superset: the original ``lithops_job_tracker.py``
    plus the extended fields from ``lithops_s1_rtc_orchestrator.py``.
    """

    job_id: str
    task_type: str  # 'DEM', 'S1', 'S2', 'S1_RTC', 'PREDICT'
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
    # S1 RTC / partial-success fields
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
