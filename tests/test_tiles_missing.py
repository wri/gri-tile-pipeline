import os
import pytest
import tempfile
import pandas as pd
from click.testing import CliRunner
from gri_tile_pipeline.cli import gri_ttc  # the module exposing the group
from tests.constants import REPO_ROOT
#
GEOPARQUET = os.path.join(REPO_ROOT, "temp/tm.geoparquet")
config_file = os.path.join(REPO_ROOT, "config.yaml")
tiledb_file = os.path.join(REPO_ROOT, "data", "tiledb.parquet")

def test_tiles_missing_with_baseline():
    outermost_project_phase_name = "BASELINE"
    project_short_name = 'TEST_01_GRI'; delim_ids = '22560785-182b-4c06-b5a5-f355eaf4f907'

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_csv_file = os.path.join(tmpdirname, f"{project_short_name}_missing_tiles.csv")

        args = [
            "--config", config_file,  # group-level option (before subcommand)
            "tiles",  # subcommand
            "missing",  # sub-subcommand
            "--geoparquet", GEOPARQUET,
            "--tiledb", tiledb_file,
            "--outermost_project_phase_name", outermost_project_phase_name,
            "--short-name", project_short_name,
            "-o", temp_csv_file,
            "--delimited_polygon_ids", delim_ids,
        ]
        # noinspection PyTypeChecker
        result = runner.invoke(gri_ttc, args, catch_exceptions=True)

        assert result.exit_code == 0
        assert os.path.isfile(temp_csv_file)
        tiles = pd.read_csv(filepath_or_buffer=temp_csv_file)
        assert len(tiles) == 2
        assert tiles['Year'][0] == 2020
        assert tiles['Year'][1] == 2021


def test_tiles_missing_with_early_insights():
    outermost_project_phase_name = "EARLY_INSIGHTS"
    project_short_name = 'TEST_01_GRI'; delim_ids = '22560785-182b-4c06-b5a5-f355eaf4f907'

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_csv_file = os.path.join(tmpdirname, f"{project_short_name}_missing_tiles.csv")

        args = [
            "--config", config_file,  # group-level option (before subcommand)
            "tiles",  # subcommand
            "missing",  # sub-subcommand
            "--geoparquet", GEOPARQUET,
            "--tiledb", tiledb_file,
            "--outermost_project_phase_name", outermost_project_phase_name,
            "--short-name", project_short_name,
            "-o", temp_csv_file,
            "--delimited_polygon_ids", delim_ids,
        ]
        # noinspection PyTypeChecker
        result = runner.invoke(gri_ttc, args, catch_exceptions=True)

        assert result.exit_code == 0
        assert os.path.isfile(temp_csv_file)
        tiles = pd.read_csv(filepath_or_buffer=temp_csv_file)
        assert len(tiles) == 5
        assert tiles['Year'][0] == 2020
        assert tiles['Year'][1] == 2021


def test_tiles_missing_with_early_endline():
    outermost_project_phase_name = "ENDLINE"
    project_short_name = 'TEST_01_GRI'; delim_ids = '22560785-182b-4c06-b5a5-f355eaf4f907'

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_csv_file = os.path.join(tmpdirname, f"{project_short_name}_missing_tiles.csv")

        args = [
            "--config", config_file,  # group-level option (before subcommand)
            "tiles",  # subcommand
            "missing",  # sub-subcommand
            "--geoparquet", GEOPARQUET,
            "--tiledb", tiledb_file,
            "--outermost_project_phase_name", outermost_project_phase_name,
            "--short-name", project_short_name,
            "-o", temp_csv_file,
            "--delimited_polygon_ids", delim_ids,
        ]
        # noinspection PyTypeChecker
        result = runner.invoke(gri_ttc, args, catch_exceptions=True)

        assert result.exit_code == 0
        assert os.path.isfile(temp_csv_file)
        tiles = pd.read_csv(filepath_or_buffer=temp_csv_file)
        # Note: Execution in year 2026 will yield 6 rows, whereas later years will yield a greater number of rows
        assert len(tiles) >= 6
        assert tiles['Year'][0] == 2020
        assert tiles['Year'][1] == 2021
