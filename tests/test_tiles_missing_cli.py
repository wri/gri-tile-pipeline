"""Tests for the ``gri-ttc tiles missing`` Click command (``tiles_missing``).

The command imports its collaborators *inside the function body*, e.g.::

    from gri_tile_pipeline.tiles.missing import generate_missing_tiles, summarize_missing

so the patch targets below are the **source** modules where those names are
defined, not the module that defines the command. That is what makes the
``unittest.mock.patch`` targets work regardless of how the command imports them.

NOTE - adjust these two import lines to your project's real layout:
  * ``tiles_missing`` / ``ExitCode`` are imported from the module that defines them.
  * The patch-target strings are the modules the command imports *from*.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

# --- adjust to the module that defines the command + ExitCode -----------------
from gri_tile_pipeline.cli import tiles_missing, ExitCode

# --- patch targets: the SOURCE modules the command imports from ---------------
P_SUMMARIZE = "gri_tile_pipeline.tiles.missing.summarize_missing"
P_GENERATE = "gri_tile_pipeline.tiles.missing.generate_missing_tiles"
P_WRITE_CSV = "gri_tile_pipeline.tiles.csv_io.write_tiles_csv"
P_EMIT_JSON = "gri_tile_pipeline.cli_context.emit_json"
P_GET_CTX = "gri_tile_pipeline.cli_context.get"


@pytest.fixture
def runner():
    return CliRunner()


def _make_gri(*, json_mode=False, parquet_path="config/tiles.parquet"):
    """Fake gri context object returned by ``get_ctx(ctx)``."""
    gri = MagicMock()
    gri.json_mode = json_mode
    gri.cfg.parquet_path = parquet_path
    return gri


SUMMARY_FIXTURE = {
    "total_missing": 1234,
    "by_cohort": [
        {"framework_key": "hbf", "polygons_missing_ttc": 800},
        {"framework_key": None, "polygons_missing_ttc": 434},
    ],
    "by_project": [
        {"short_name": "RWA_23_AEE", "framework_key": "hbf", "polygons_missing_ttc": 800},
    ],
}

TILES_FIXTURE = [
    {"year": 2022, "lon": 1.0, "lat": 2.0, "X_tile": 10, "Y_tile": 20},
    {"year": 2022, "lon": 1.1, "lat": 2.1, "X_tile": 11, "Y_tile": 21},
    {"year": 2023, "lon": 1.2, "lat": 2.2, "X_tile": 12, "Y_tile": 22},
]


# --------------------------------------------------------------------------- #
# Summary branch: no --output, no --short-name, no --framework-key
# --------------------------------------------------------------------------- #
def test_summary_branch_calls_summarize_and_prints(runner):
    with patch(P_GET_CTX, return_value=_make_gri(json_mode=False)), \
         patch(P_SUMMARIZE, return_value=SUMMARY_FIXTURE) as m_sum, \
         patch(P_GENERATE) as m_gen, \
         patch(P_EMIT_JSON) as m_emit:
        result = runner.invoke(tiles_missing, [])

    assert result.exit_code == 0, result.output
    # summarize called with positional (geoparquet, phase); generate NOT called
    m_sum.assert_called_once()
    assert m_sum.call_args.args[1] == "BASELINE"
    m_gen.assert_not_called()
    # human-readable output rendered
    assert "Total polygons missing TTC: 1,234" in result.output
    assert "hbf" in result.output
    assert "(null)" in result.output  # None framework_key falls back to '(null)'
    # emit_json got the summary payload
    payload = m_emit.call_args.args[1]
    assert payload["command"] == "tiles.missing"
    assert payload["status"] == "summary"
    assert payload["total_missing"] == 1234


def test_summary_branch_json_mode_suppresses_echo(runner):
    with patch(P_GET_CTX, return_value=_make_gri(json_mode=True)), \
         patch(P_SUMMARIZE, return_value=SUMMARY_FIXTURE), \
         patch(P_EMIT_JSON) as m_emit:
        result = runner.invoke(tiles_missing, [])

    assert result.exit_code == 0, result.output
    # json_mode True => no human echo
    assert "Total polygons missing TTC" not in result.output
    m_emit.assert_called_once()


def test_summary_uses_phase_option(runner):
    with patch(P_GET_CTX, return_value=_make_gri()), \
         patch(P_SUMMARIZE, return_value=SUMMARY_FIXTURE) as m_sum, \
         patch(P_EMIT_JSON):
        result = runner.invoke(
            tiles_missing, ["--outermost_project_phase_name", "ENDLINE"]
        )

    assert result.exit_code == 0, result.output
    assert m_sum.call_args.args[1] == "ENDLINE"


# --------------------------------------------------------------------------- #
# Generate branch: triggered by --output OR a filter option
# --------------------------------------------------------------------------- #
def test_generate_with_output_writes_csv_and_emits_counts(runner):
    gri = _make_gri(parquet_path="cfg/default.parquet")
    with patch(P_GET_CTX, return_value=gri), \
         patch(P_GENERATE, return_value=TILES_FIXTURE) as m_gen, \
         patch(P_WRITE_CSV) as m_write, \
         patch(P_SUMMARIZE) as m_sum, \
         patch(P_EMIT_JSON) as m_emit:
        result = runner.invoke(tiles_missing, ["-o", "out.csv"])

    assert result.exit_code == 0, result.output
    m_sum.assert_not_called()
    m_gen.assert_called_once()
    # tiledb defaulted to config parquet_path
    assert m_gen.call_args.args[1] == "cfg/default.parquet"
    # csv written with the produced tiles
    m_write.assert_called_once_with("out.csv", TILES_FIXTURE)
    # by_year counts aggregated correctly
    payload = m_emit.call_args.args[1]
    assert payload["status"] == "ok"
    assert payload["n_tiles"] == 3
    assert payload["output"] == "out.csv"
    assert payload["by_year"] == {2022: 2, 2023: 1}


def test_filter_option_triggers_generate_without_output(runner):
    """--short-name alone (no -o) must still hit the generate branch, not summary."""
    with patch(P_GET_CTX, return_value=_make_gri()), \
         patch(P_GENERATE, return_value=TILES_FIXTURE) as m_gen, \
         patch(P_WRITE_CSV) as m_write, \
         patch(P_SUMMARIZE) as m_sum, \
         patch(P_EMIT_JSON) as m_emit:
        result = runner.invoke(tiles_missing, ["--short-name", "RWA_23_AEE"])

    assert result.exit_code == 0, result.output
    m_sum.assert_not_called()
    m_gen.assert_called_once()
    assert m_gen.call_args.kwargs["short_name"] == "RWA_23_AEE"
    # no --output => csv not written, but counts still emitted
    m_write.assert_not_called()
    assert m_emit.call_args.args[1]["status"] == "ok"


def test_explicit_tiledb_overrides_config(runner):
    with patch(P_GET_CTX, return_value=_make_gri(parquet_path="cfg/default.parquet")), \
         patch(P_GENERATE, return_value=TILES_FIXTURE) as m_gen, \
         patch(P_WRITE_CSV), patch(P_EMIT_JSON):
        result = runner.invoke(
            tiles_missing, ["--tiledb", "explicit/tiles.parquet", "-o", "out.csv"]
        )

    assert result.exit_code == 0, result.output
    assert m_gen.call_args.args[1] == "explicit/tiles.parquet"


def test_framework_key_passed_through(runner):
    with patch(P_GET_CTX, return_value=_make_gri()), \
         patch(P_GENERATE, return_value=TILES_FIXTURE) as m_gen, \
         patch(P_WRITE_CSV), patch(P_EMIT_JSON):
        result = runner.invoke(
            tiles_missing, ["--framework-key", "hbf", "-o", "out.csv"]
        )

    assert result.exit_code == 0, result.output
    assert m_gen.call_args.kwargs["framework_key"] == "hbf"


# --------------------------------------------------------------------------- #
# delimited_polygon_ids parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("a,b,c", ["a", "b", "c"]),
        (" a , b , c ", ["a", "b", "c"]),          # whitespace stripped
        ("'a','b'", ["a", "b"]),                    # surrounding single quotes stripped
        ("'a', 'b' ", ["a", "b"]),                  # mixed spacing + quotes
    ],
)
def test_delimited_polygon_ids_parsing(runner, raw, expected):
    with patch(P_GET_CTX, return_value=_make_gri()), \
         patch(P_GENERATE, return_value=TILES_FIXTURE) as m_gen, \
         patch(P_WRITE_CSV), patch(P_EMIT_JSON):
        result = runner.invoke(
            tiles_missing, ["--delimited_polygon_ids", raw, "-o", "out.csv"]
        )

    assert result.exit_code == 0, result.output
    assert m_gen.call_args.kwargs["polygon_ids"] == expected


def test_polygon_ids_none_when_option_absent(runner):
    with patch(P_GET_CTX, return_value=_make_gri()), \
         patch(P_GENERATE, return_value=TILES_FIXTURE) as m_gen, \
         patch(P_WRITE_CSV), patch(P_EMIT_JSON):
        result = runner.invoke(tiles_missing, ["-o", "out.csv"])

    assert result.exit_code == 0, result.output
    assert m_gen.call_args.kwargs["polygon_ids"] is None


# --------------------------------------------------------------------------- #
# No-work branch: generate returns empty
# --------------------------------------------------------------------------- #
def test_no_work_exits_with_no_work_code(runner):
    with patch(P_GET_CTX, return_value=_make_gri()), \
         patch(P_GENERATE, return_value=[]), \
         patch(P_WRITE_CSV) as m_write, \
         patch(P_EMIT_JSON) as m_emit:
        result = runner.invoke(tiles_missing, ["-o", "out.csv"])

    assert result.exit_code == ExitCode.NO_WORK
    m_write.assert_not_called()
    payload = m_emit.call_args.args[1]
    assert payload["status"] == "no_work"
    assert payload["n_tiles"] == 0
