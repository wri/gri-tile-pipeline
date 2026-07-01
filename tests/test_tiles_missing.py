import os
import pytest
import tempfile
import pandas as pd
from click.testing import CliRunner
from gri_shared_library.constants import TERRAMATCH_GEOPARQUET_FILEPATH, TTC_TILEDB_FILEPATH
from gri_tile_pipeline.cli import gri_ttc  # the module exposing the group
from tests.constants import REPO_ROOT
#
GEOPARQUET = TERRAMATCH_GEOPARQUET_FILEPATH
tiledb_file = TTC_TILEDB_FILEPATH
config_file = os.path.join(REPO_ROOT, "config.yaml")

pytestmark = pytest.mark.skipif(
    not os.getenv("AWS_PROFILE"),
    reason="requires AWS_PROFILE secret",
)
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
        assert len(tiles) == 1
        assert tiles['Year'][0] == 2021


pytestmark = pytest.mark.skipif(
    not os.getenv("AWS_PROFILE"),
    reason="requires AWS_PROFILE secret",
)
def test_tiles_missing_with_early_insight():
    outermost_project_phase_name = "EARLY_INSIGHT"
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
        assert tiles['Year'][0] == 2021
        assert tiles['Year'][1] == 2024


pytestmark = pytest.mark.skipif(
    not os.getenv("AWS_PROFILE"),
    reason="requires AWS_PROFILE secret",
)
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
        assert len(tiles) >= 2
        assert tiles['Year'][0] == 2021
        assert tiles['Year'][1] == 2024
