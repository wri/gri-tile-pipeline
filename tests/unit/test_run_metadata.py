"""Unit tests for tracking/run_metadata.py."""

import pytest

from gri_tile_pipeline.tracking.run_metadata import PipelineRun, StepResult, TileStatus


class TestPipelineRun:
    def test_ensure_tile_creates(self):
        run = PipelineRun(
            run_id="test-001",
            started_at="2024-01-01T00:00:00Z",
            config_snapshot={},
            dest="s3://bucket/prefix",
        )
        ts = run.ensure_tile(1000, 871, 2024, -73.5, 45.5)
        assert isinstance(ts, TileStatus)
        assert ts.x_tile == 1000
        assert ts.y_tile == 871
        assert "1000X871Y" in run.tiles

    def test_ensure_tile_idempotent(self):
        run = PipelineRun(
            run_id="test-001",
            started_at="2024-01-01T00:00:00Z",
            config_snapshot={},
            dest="s3://bucket/prefix",
        )
        ts1 = run.ensure_tile(1000, 871, 2024, -73.5, 45.5)
        ts2 = run.ensure_tile(1000, 871, 2024, -73.5, 45.5)
        assert ts1 is ts2
        assert len(run.tiles) == 1

    def test_tiles_pending_step_all_pending(self):
        run = PipelineRun(
            run_id="test-001",
            started_at="2024-01-01T00:00:00Z",
            config_snapshot={},
            dest="s3://bucket/prefix",
        )
        run.ensure_tile(1000, 871, 2024, -73.5, 45.5)
        run.ensure_tile(1001, 872, 2024, -73.6, 45.6)

        pending = run.tiles_pending_step("download_ard")
        assert len(pending) == 2

    def test_tiles_pending_step_some_done(self):
        run = PipelineRun(
            run_id="test-001",
            started_at="2024-01-01T00:00:00Z",
            config_snapshot={},
            dest="s3://bucket/prefix",
        )
        ts1 = run.ensure_tile(1000, 871, 2024, -73.5, 45.5)
        ts1.steps["download_ard"] = StepResult(status="success")
        run.ensure_tile(1001, 872, 2024, -73.6, 45.6)

        pending = run.tiles_pending_step("download_ard")
        assert len(pending) == 1
        assert "1001X872Y" in pending

    def test_tiles_pending_step_skipped_excluded(self):
        run = PipelineRun(
            run_id="test-001",
            started_at="2024-01-01T00:00:00Z",
            config_snapshot={},
            dest="s3://bucket/prefix",
        )
        ts = run.ensure_tile(1000, 871, 2024, -73.5, 45.5)
        ts.steps["predict"] = StepResult(status="skipped")

        pending = run.tiles_pending_step("predict")
        assert len(pending) == 0

    def test_tile_key(self):
        run = PipelineRun(
            run_id="test-001",
            started_at="2024-01-01T00:00:00Z",
            config_snapshot={},
            dest="s3://bucket/prefix",
        )
        assert run.tile_key(1000, 871) == "1000X871Y"
