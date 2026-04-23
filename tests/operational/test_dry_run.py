"""Dry-run tests: commands with --dry-run should not perform actual work."""

import os

import pytest

from gri_tile_pipeline.tiles.csv_io import write_tiles_csv


@pytest.fixture
def sample_tiles_csv(tmp_path):
    """Create a minimal tiles CSV for dry-run testing."""
    tiles = [
        {"year": 2024, "lon": -73.5, "lat": 45.5, "X_tile": 1000, "Y_tile": 871},
    ]
    csv_path = str(tmp_path / "tiles.csv")
    write_tiles_csv(csv_path, tiles)
    return csv_path


def test_download_dry_run(sample_tiles_csv):
    """download --dry-run --local should preview without executing."""
    from click.testing import CliRunner

    from gri_tile_pipeline.cli import gri_ttc

    runner = CliRunner()
    result = runner.invoke(
        gri_ttc,
        ["download", sample_tiles_csv, "--dest", "/tmp/test", "--dry-run", "--local"],
    )
    assert result.exit_code == 0
    assert "Tiles: 1" in result.output
    assert "local" in result.output.lower()


def test_predict_dry_run(sample_tiles_csv):
    """predict --dry-run --local should preview without executing."""
    from click.testing import CliRunner

    from gri_tile_pipeline.cli import gri_ttc

    runner = CliRunner()
    result = runner.invoke(
        gri_ttc,
        ["predict", sample_tiles_csv, "--dest", "/tmp/test", "--dry-run", "--local"],
    )
    assert result.exit_code == 0
    assert "Tiles: 1" in result.output


def test_cost_command(sample_tiles_csv):
    """cost command should show dollar amounts."""
    from click.testing import CliRunner

    from gri_tile_pipeline.cli import gri_ttc

    runner = CliRunner()
    result = runner.invoke(gri_ttc, ["cost", sample_tiles_csv])
    assert result.exit_code == 0
    assert "$" in result.output
