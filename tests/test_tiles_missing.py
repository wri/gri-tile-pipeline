import os
import tempfile
import pandas as pd
from click.testing import CliRunner
from gri_tile_pipeline.cli import gri_ttc  # the module exposing the group
from tests.constants import REPO_ROOT

GEOPARQUET = os.path.join(REPO_ROOT, "temp/tm.geoparquet")
config_file = os.path.join(REPO_ROOT, "config.yaml")
# CFG = load_config(config_file)

def test_tiles_missing_epoch():
    tiledb_file = os.path.join(REPO_ROOT, "data", "tiledb.parquet")
    framework_key = 'terrafund-landscapes'

    project_short_name = 'TEST_01_GRI'; delim_ids = '22560785-182b-4c06-b5a5-f355eaf4f907'
    # project_short_name = 'RWA_22_ARCOS'
    # project_short_name = 'BUR_23_RB2000'

    """Invoke `gri-ttc resolve` programmatically. Returns click.testing.Result."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_csv_file = os.path.join(tmpdirname, f"{project_short_name}_missing_tiles.csv")

        args = [
            "--config", config_file,  # group-level option (before subcommand)
            "tiles",  # subcommand
            "missing",  # sub-subcommand
            "--geoparquet", GEOPARQUET,
            "--tiledb", tiledb_file,
            # "--eval_epoch", "baseline"
            "--eval_epoch", "midway",
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
        assert tiles['Year'][0] == 2024
