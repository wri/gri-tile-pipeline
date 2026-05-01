import os
import tempfile

import pandas as pd
import pytest

from gri_tile_pipeline.config import load_config
from gri_tile_pipeline.steps.project_e2e import run_project_pipeline
from gri_tile_pipeline.tiles.missing import generate_missing_tiles, summarize_missing
from gri_tile_pipeline.cli import run_project, cost_function, resolve_function, tiles_missing_function
from tests.conftest import REPO_ROOT


# Best guide for using the repo: https://github.com/wri/gri-tile-pipeline/blob/develop/docs/cli_workflows.md


ROOT_PATH = REPO_ROOT
GEOPARQUET_FILE = os.path.join(ROOT_PATH, "temp", "tm.geoparquet")
config_file = os.path.join(ROOT_PATH, "config.yaml")
CFG = load_config(config_file)

def test_missing_tiles():
    tiledb_file = os.path.join(ROOT_PATH, "data", "tiledb.parquet")
    framework_key = 'terrafund-landscapes'

    project_short_name = 'TEST_01_GRI'
    # project_short_name = 'RWA_22_ARCOS'

    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_csv_file = os.path.join(tmpdirname, f"{project_short_name}_missing_tiles.csv")
        results = tiles_missing_function(CFG, geoparquet=GEOPARQUET_FILE, tiledb=tiledb_file, short_name=project_short_name,
                                     framework_key=framework_key, output=temp_csv_file)

        if results['n_tiles'] > 0:
            missing_tiles = pd.read_csv(filepath_or_buffer=temp_csv_file)
        else:
            missing_tiles = pd.DataFrame()

    assert len(missing_tiles) == results['n_tiles']


def test_summarize_missing_ttc():
    project_short_name = 'TEST_01_GRI'

    summary = summarize_missing(GEOPARQUET_FILE)

    summary_row = [s for s in summary['by_project'] if s['short_name'] == project_short_name]
    polys_missing_ttc = summary_row[0]['polygons_missing_ttc']

    assert polys_missing_ttc == 3


# def test_project():
#     project_short_name = 'TEST_01_GRI'
#
#     results = run_project_pipeline(short_name=project_short_name, dest='/home/kennc/', cfg=CFG, check_only=True)
#
#     b=2

def test_cost():
    project_short_name = 'TEST_01_GRI'
    sample_poly_uuids = ['22560785-182b-4c06-b5a5-f355eaf4f907','a09e8e9f-f0f6-438a-95ac-aa5457082d46']
    poly_uuids_str = ",".join(f"'{v}'" for v in sample_poly_uuids)
    where_sql = f"poly_uuid in ({poly_uuids_str})"

    # Create a sample tiles.csv
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_csv_file = os.path.join(tmpdirname, f"{project_short_name}_tiles.csv")
        resolve_function(CFG, input=None, output=temp_csv_file,
                         where_sql=where_sql)

        results = cost_function(CFG, temp_csv_file, mem=None, include_predict=True)

    assert results['predict_cost'] == pytest.approx(0.018, 0.01)


def test_resolve():
    project_short_name = 'TEST_01_GRI'

    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_csv_file = os.path.join(tmpdirname, f"{project_short_name}_tiles.csv")
        results = resolve_function(CFG, input=project_short_name, output=temp_csv_file)

        assert os.path.isfile(temp_csv_file)
        tiles = pd.read_csv(filepath_or_buffer=temp_csv_file)
        assert len(tiles) == results['n_tiles']

    assert results['n_tiles'] == 1
    assert results['metadata']['n_polygons'] == 3
    assert results['metadata']['n_missing_ttc'] == 3



