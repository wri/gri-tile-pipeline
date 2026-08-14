"""Unit tests for the polygons-missing-ttc CLI command."""

from unittest.mock import MagicMock, call, patch

import pytest
from click.testing import CliRunner

# Adjust this import to match your actual module path
from gri_tile_pipeline.cli import gri_ttc


MOCK_POLYGONS = [
    {"polygon_id": "poly-1", "eval_year": 2023},
    {"polygon_id": "poly-2", "eval_year": 2023},
    {"polygon_id": "poly-3", "eval_year": 2024},
]

PATCH_LIST = "gri_tile_pipeline.tiles.missing.list_polygons_missing_ttc"
PATCH_WRITE = "gri_tile_pipeline.tiles.csv_io.write_polygons_csv"
PATCH_EMIT = "gri_tile_pipeline.cli_context.emit_json"


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def invoke(runner, args, catch_exceptions=False):
    return runner.invoke(gri_ttc, ["polygons-missing-ttc"] + args, catch_exceptions=catch_exceptions)


# ---------------------------------------------------------------------------
# Required --output flag
# ---------------------------------------------------------------------------

def test_missing_output_flag_fails(runner):
    result = invoke(runner, [])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "--output" in result.output


# ---------------------------------------------------------------------------
# Normal (ok) path
# ---------------------------------------------------------------------------

@patch(PATCH_EMIT)
@patch(PATCH_WRITE)
@patch(PATCH_LIST, return_value=MOCK_POLYGONS)
def test_ok_writes_csv_and_emits(mock_list, mock_write, mock_emit, runner):
    result = invoke(runner, ["-o", "out.csv"])

    assert result.exit_code == 0
    mock_list.assert_called_once_with(
        "temp/tm.geoparquet",
        outermost_project_phase_name="BASELINE",
        short_name=None,
        framework_key=None,
        polygon_ids=None,
    )
    mock_write.assert_called_once_with("out.csv", MOCK_POLYGONS)
    _, emitted = mock_emit.call_args[0]  # (ctx, payload)
    assert emitted["status"] == "ok"
    assert emitted["n_polygons"] == 3
    assert emitted["output"] == "out.csv"
    assert emitted["by_year"] == {2023: 2, 2024: 1}


# ---------------------------------------------------------------------------
# No-work path
# ---------------------------------------------------------------------------

@patch(PATCH_EMIT)
@patch(PATCH_LIST, return_value=[])
def test_no_polygons_emits_no_work(mock_list, mock_emit, runner):
    result = invoke(runner, ["-o", "out.csv"])

    # ExitCode.NO_WORK should be non-zero; exact value depends on your enum
    assert result.exit_code != 0
    _, emitted = mock_emit.call_args[0]
    assert emitted["status"] == "no_work"
    assert emitted["n_polygons"] == 0


# ---------------------------------------------------------------------------
# Filter options are forwarded correctly
# ---------------------------------------------------------------------------

@patch(PATCH_EMIT)
@patch(PATCH_WRITE)
@patch(PATCH_LIST, return_value=MOCK_POLYGONS)
def test_short_name_forwarded(mock_list, mock_write, mock_emit, runner):
    invoke(runner, ["--short-name", "RWA_23_AEE", "-o", "out.csv"])

    mock_list.assert_called_once_with(
        "temp/tm.geoparquet",
        outermost_project_phase_name="BASELINE",
        short_name="RWA_23_AEE",
        framework_key=None,
        polygon_ids=None,
    )


@patch(PATCH_EMIT)
@patch(PATCH_WRITE)
@patch(PATCH_LIST, return_value=MOCK_POLYGONS)
def test_framework_key_forwarded(mock_list, mock_write, mock_emit, runner):
    invoke(runner, ["--framework-key", "hbf", "-o", "out.csv"])

    mock_list.assert_called_once_with(
        "temp/tm.geoparquet",
        outermost_project_phase_name="BASELINE",
        short_name=None,
        framework_key="hbf",
        polygon_ids=None,
    )


@patch(PATCH_EMIT)
@patch(PATCH_WRITE)
@patch(PATCH_LIST, return_value=MOCK_POLYGONS)
def test_project_phase_forwarded(mock_list, mock_write, mock_emit, runner):
    invoke(runner, ["--outermost_project_phase_name", "ENDLINE", "-o", "out.csv"])

    mock_list.assert_called_once_with(
        "temp/tm.geoparquet",
        outermost_project_phase_name="ENDLINE",
        short_name=None,
        framework_key=None,
        polygon_ids=None,
    )


@patch(PATCH_EMIT)
@patch(PATCH_WRITE)
@patch(PATCH_LIST, return_value=MOCK_POLYGONS)
def test_custom_geoparquet_path(mock_list, mock_write, mock_emit, runner):
    invoke(runner, ["--geoparquet", "data/custom.geoparquet", "-o", "out.csv"])

    mock_list.assert_called_once_with(
        "data/custom.geoparquet",
        outermost_project_phase_name="BASELINE",
        short_name=None,
        framework_key=None,
        polygon_ids=None,
    )


# ---------------------------------------------------------------------------
# Hidden --delimited_polygon_ids parsing
# ---------------------------------------------------------------------------

@patch(PATCH_EMIT)
@patch(PATCH_WRITE)
@patch(PATCH_LIST, return_value=MOCK_POLYGONS)
def test_delimited_polygon_ids_parsed(mock_list, mock_write, mock_emit, runner):
    invoke(runner, ["--delimited_polygon_ids", "id-1, 'id-2', id-3", "-o", "out.csv"])

    mock_list.assert_called_once_with(
        "temp/tm.geoparquet",
        outermost_project_phase_name="BASELINE",
        short_name=None,
        framework_key=None,
        polygon_ids=["id-1", "id-2", "id-3"],
    )


@patch(PATCH_EMIT)
@patch(PATCH_WRITE)
@patch(PATCH_LIST, return_value=MOCK_POLYGONS)
def test_single_polygon_id(mock_list, mock_write, mock_emit, runner):
    invoke(runner, ["--delimited_polygon_ids", "only-one", "-o", "out.csv"])

    _, kwargs = mock_list.call_args
    assert kwargs["polygon_ids"] == ["only-one"]


# ---------------------------------------------------------------------------
# Year aggregation in emitted payload
# ---------------------------------------------------------------------------

@patch(PATCH_EMIT)
@patch(PATCH_WRITE)
@patch(PATCH_LIST, return_value=[
    {"polygon_id": "a", "eval_year": 2022},
    {"polygon_id": "b", "eval_year": 2022},
    {"polygon_id": "c", "eval_year": 2022},
])
def test_by_year_counts(mock_list, mock_write, mock_emit, runner):
    invoke(runner, ["-o", "out.csv"])

    _, emitted = mock_emit.call_args[0]
    assert emitted["by_year"] == {2022: 3}