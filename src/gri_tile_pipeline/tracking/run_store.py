"""JSON-file persistence for PipelineRun metadata."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from gri_tile_pipeline.tracking.run_metadata import (
    PipelineRun,
    StepResult,
    TileStatus,
)


class RunStore:
    """Persist :class:`PipelineRun` to local JSON files in a base directory."""

    def __init__(self, base_dir: str = ".gri_runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.base_dir / f"{run_id}.json"

    def save(self, run: PipelineRun) -> Path:
        """Write *run* to ``<base_dir>/<run_id>.json``."""
        path = self._path(run.run_id)
        with open(path, "w") as f:
            json.dump(asdict(run), f, indent=2, default=str)
        logger.debug(f"Run metadata saved to {path}")
        return path

    def load(self, run_id: str) -> PipelineRun:
        """Load a PipelineRun from disk."""
        path = self._path(run_id)
        with open(path) as f:
            data = json.load(f)

        tiles = {}
        for key, td in data.pop("tiles", {}).items():
            steps = {}
            for step_name, sr in td.pop("steps", {}).items():
                steps[step_name] = StepResult(**sr)
            tiles[key] = TileStatus(**td, steps=steps)

        return PipelineRun(**data, tiles=tiles)

    def list_runs(self) -> list[str]:
        """Return run IDs of all saved runs (newest first)."""
        paths = sorted(self.base_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        return [p.stem for p in paths]

    def update_tile(
        self, run_id: str, tile_key: str, step: str, result: StepResult,
    ) -> None:
        """Update a single tile's step result and re-save."""
        run = self.load(run_id)
        if tile_key in run.tiles:
            run.tiles[tile_key].steps[step] = result
        self.save(run)
