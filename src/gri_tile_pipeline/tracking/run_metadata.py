"""Dataclasses for pipeline run metadata and per-tile status tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    """Result of one pipeline step for one tile."""

    status: str = "pending"  # "pending" | "success" | "failed" | "skipped"
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    error: str | None = None
    output_keys: list[str] | None = None  # S3 keys or local paths produced


@dataclass
class TileStatus:
    """Per-tile status across all pipeline steps."""

    x_tile: int
    y_tile: int
    year: int
    lon: float
    lat: float
    steps: dict[str, StepResult] = field(default_factory=dict)


@dataclass
class PipelineRun:
    """Top-level run metadata."""

    run_id: str
    started_at: str
    config_snapshot: dict[str, Any]
    dest: str
    tiles: dict[str, TileStatus] = field(default_factory=dict)  # "XtileXYtileY" -> status
    current_step: str | None = None
    status: str = "running"  # "running" | "completed" | "failed" | "partial"
    finished_at: str | None = None

    def tile_key(self, x_tile: int, y_tile: int) -> str:
        return f"{x_tile}X{y_tile}Y"

    def ensure_tile(self, x_tile: int, y_tile: int, year: int, lon: float, lat: float) -> TileStatus:
        """Get or create a TileStatus entry."""
        key = self.tile_key(x_tile, y_tile)
        if key not in self.tiles:
            self.tiles[key] = TileStatus(
                x_tile=x_tile, y_tile=y_tile, year=year, lon=lon, lat=lat,
            )
        return self.tiles[key]

    def tiles_pending_step(self, step_name: str) -> list[str]:
        """Return tile keys that haven't succeeded for *step_name*."""
        pending = []
        for key, ts in self.tiles.items():
            step = ts.steps.get(step_name)
            if step is None or step.status not in ("success", "skipped"):
                pending.append(key)
        return pending
