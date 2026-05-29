"""Tests for the ``tiles missing`` CLI command and its eval_epoch handling.

Covers:
  - CLI routes eval_epoch correctly to generate_missing_tiles()
  - All three epoch names (BASELINE / MIDWAY / ENDLINE) map to the right offsets
  - Default epoch is BASELINE
  - Invalid epoch name raises ValueError inside generate_missing_tiles
  - NO_WORK exit path when generate_missing_tiles returns an empty list
  - Summary-only path (no --short-name / --framework-key / --output)
  - Output written when --output is supplied
  - framework_key filter forwarded correctly
  - Combined eval_epoch + framework_key forwarding
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from gri_tile_pipeline.cli import gri_ttc
from gri_shared_library.constants import EvalEpoch
from gri_tile_pipeline.tiles.missing import generate_missing_tiles


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAKE_TILE = {"year": 2022, "lon": 10.0, "lat": 5.0, "X_tile": 100, "Y_tile": 50}

# Patch targets — use the *defining* module (lazy imports inside the CLI
# function bind to the original location, not via gri_tile_pipeline.cli).
_GEN_MISSING = "gri_tile_pipeline.tiles.missing.generate_missing_tiles"
_SUMMARIZE   = "gri_tile_pipeline.tiles.missing.summarize_missing"
_WRITE_CSV   = "gri_tile_pipeline.tiles.csv_io.write_tiles_csv"
_LOAD_CFG    = "gri_tile_pipeline.config.load_config"
_SETUP_LOG   = "gri_tile_pipeline.logging.setup_logging"
_BIND_CTX    = "gri_tile_pipeline.logging.bind_run_context"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _stub_infra():
    """Suppress logging setup and config file I/O for every test in this file."""
    from gri_tile_pipeline.config import PipelineConfig
    stub_cfg = PipelineConfig()  # all defaults; parquet_path='data/tiledb.parquet'

    with (
        patch(_LOAD_CFG, return_value=stub_cfg),
        patch(_SETUP_LOG),
        patch(_BIND_CTX),
        patch("gri_tile_pipeline.cli_context.resolve_pipeline_version", return_value="test"),
        patch("gri_tile_pipeline.cli_context.resolve_git_sha", return_value="abc123"),
    ):
        yield


def invoke_missing(runner, extra_args: list[str]):
    """Run ``gri-ttc tiles missing`` with the given extra arguments."""
    return runner.invoke(gri_ttc, ["tiles", "missing", *extra_args],
                         catch_exceptions=False)


# ---------------------------------------------------------------------------
# Unit tests: generate_missing_tiles — eval_epoch → survey_year_offset mapping
# ---------------------------------------------------------------------------

class TestOutermostEvalEpochOffsetMapping:
    """Verify each epoch name drives the correct $2 parameter in the SQL query."""

    def _captured_offset(self, epoch) -> int:
        """Return the survey_year_offset ($2) that the DuckDB execute() receives."""
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = []

        with patch("gri_tile_pipeline.tiles.missing.connect_with_spatial",
                   return_value=mock_con):
            generate_missing_tiles(
                geoparquet="fake.parquet",
                tiledb="fake_tiledb.parquet",
                outermost_eval_epoch=epoch,
            )

        # params = second positional arg to con.execute(query, params)
        params: list = mock_con.execute.call_args[0][1]
        return params[1]  # index 1 == $2 == survey_year_offset

    def test_baseline_offset_is_minus_one(self):
        assert self._captured_offset("BASELINE") == EvalEpoch.BASELINE.value  # -1

    def test_midway_offset_is_two(self):
        assert self._captured_offset("MIDWAY") == EvalEpoch.MIDWAY.value      # 2

    def test_endline_offset_is_six(self):
        assert self._captured_offset("ENDLINE") == EvalEpoch.ENDLINE.value    # 6

    def test_lowercase_baseline(self):
        assert self._captured_offset("baseline") == EvalEpoch.BASELINE.value

    def test_lowercase_midway(self):
        assert self._captured_offset("midway") == EvalEpoch.MIDWAY.value

    def test_none_defaults_to_baseline(self):
        """eval_epoch=None is treated as BASELINE per the implementation guard."""
        assert self._captured_offset(None) == EvalEpoch.BASELINE.value

    def test_invalid_outermost_epoch_raises_value_error(self):
        mock_con = MagicMock()
        with patch("gri_tile_pipeline.tiles.missing.connect_with_spatial",
                   return_value=mock_con):
            with pytest.raises(ValueError, match="Unknown eval epoch"):
                generate_missing_tiles(
                    geoparquet="fake.parquet",
                    tiledb="fake_tiledb.parquet",
                    outermost_eval_epoch="QUARTERLY",
                )


# ---------------------------------------------------------------------------
# CLI tests: ``gri-ttc tiles missing``
# ---------------------------------------------------------------------------

class TestTilesMissingCLI:

    # -- eval_epoch forwarding -----------------------------------------------

    def test_default_outermost_eval_epoch_is_baseline(self, runner):
        with (
            patch(_GEN_MISSING, return_value=[FAKE_TILE]) as mock_gen,
            patch(_WRITE_CSV),
        ):
            result = invoke_missing(runner, ["--short-name", "TEST_23_XXX"])

        assert result.exit_code == 0, result.output
        _args, kwargs = mock_gen.call_args
        assert kwargs.get("outermost_eval_epoch", "BASELINE") == "BASELINE"

    def test_outermost_eval_epoch_baseline_forwarded(self, runner):
        with (
            patch(_GEN_MISSING, return_value=[FAKE_TILE]) as mock_gen,
            patch(_WRITE_CSV),
        ):
            result = invoke_missing(runner,
                                    ["--short-name", "TEST_23_XXX",
                                     "--outermost_eval_epoch", "BASELINE"])

        assert result.exit_code == 0, result.output
        _args, kwargs = mock_gen.call_args
        assert kwargs["outermost_eval_epoch"] == "BASELINE"

    def test_outermost_eval_epoch_midway_forwarded(self, runner):
        with (
            patch(_GEN_MISSING, return_value=[FAKE_TILE]) as mock_gen,
            patch(_WRITE_CSV),
        ):
            result = invoke_missing(runner,
                                    ["--short-name", "TEST_23_XXX",
                                     "--outermost_eval_epoch", "MIDWAY"])

        assert result.exit_code == 0, result.output
        _args, kwargs = mock_gen.call_args
        assert kwargs["outermost_eval_epoch"] == "MIDWAY"

    def test_outermost_eval_epoch_endline_forwarded(self, runner):
        with (
            patch(_GEN_MISSING, return_value=[FAKE_TILE]) as mock_gen,
            patch(_WRITE_CSV),
        ):
            result = invoke_missing(runner,
                                    ["--short-name", "TEST_23_XXX",
                                     "--outermost_eval_epoch", "ENDLINE"])

        assert result.exit_code == 0, result.output
        _args, kwargs = mock_gen.call_args
        assert kwargs["outermost_eval_epoch"] == "ENDLINE"

    # -- no-work exit --------------------------------------------------------

    def test_no_work_exit_code_when_no_tiles(self, runner):
        with (
            patch(_GEN_MISSING, return_value=[]),
            patch(_WRITE_CSV) as mock_write,
        ):
            result = invoke_missing(runner, ["--short-name", "EMPTY_PROJECT"])

        assert result.exit_code == 6  # ExitCode.NO_WORK
        mock_write.assert_not_called()

    # -- output file ---------------------------------------------------------

    def test_output_written_when_tiles_found(self, runner, tmp_path):
        out = str(tmp_path / "missing.csv")
        with (
            patch(_GEN_MISSING, return_value=[FAKE_TILE]),
            patch(_WRITE_CSV) as mock_write,
        ):
            result = invoke_missing(runner,
                                    ["--short-name", "TEST_23_XXX", "-o", out])

        assert result.exit_code == 0, result.output
        mock_write.assert_called_once_with(out, [FAKE_TILE])

    def test_no_output_flag_skips_csv_write(self, runner):
        with (
            patch(_GEN_MISSING, return_value=[FAKE_TILE]),
            patch(_WRITE_CSV) as mock_write,
        ):
            result = invoke_missing(runner, ["--short-name", "TEST_23_XXX"])

        assert result.exit_code == 0, result.output
        mock_write.assert_not_called()

    # -- summary mode --------------------------------------------------------

    def test_summary_mode_when_no_filters_or_output(self, runner):
        """No --short-name / --framework-key / --output → calls summarize_missing."""
        summary_payload = {
            "total_missing": 42,
            "by_cohort": [{"framework_key": "hbf", "polygons_missing_ttc": 42}],
            "by_project": [],
        }
        with (
            patch(_SUMMARIZE, return_value=summary_payload) as mock_sum,
            patch(_GEN_MISSING) as mock_gen,
        ):
            result = invoke_missing(runner, [])

        assert result.exit_code == 0, result.output
        mock_sum.assert_called_once()
        mock_gen.assert_not_called()
        assert "42" in result.output

    # -- filter forwarding ---------------------------------------------------

    def test_framework_key_forwarded_to_generate(self, runner):
        with (
            patch(_GEN_MISSING, return_value=[FAKE_TILE]) as mock_gen,
            patch(_WRITE_CSV),
        ):
            result = invoke_missing(runner, ["--framework-key", "hbf"])

        assert result.exit_code == 0, result.output
        _args, kwargs = mock_gen.call_args
        assert kwargs["framework_key"] == "hbf"

    # -- combined eval_epoch + framework_key ---------------------------------

    def test_midway_with_framework_key(self, runner):
        with (
            patch(_GEN_MISSING, return_value=[FAKE_TILE]) as mock_gen,
            patch(_WRITE_CSV),
        ):
            result = invoke_missing(runner,
                                    ["--framework-key", "hbf",
                                     "--outermost_eval_epoch", "MIDWAY"])

        assert result.exit_code == 0, result.output
        _args, kwargs = mock_gen.call_args
        assert kwargs["outermost_eval_epoch"] == "MIDWAY"
        assert kwargs["framework_key"] == "hbf"

    def test_endline_with_short_name(self, runner):
        with (
            patch(_GEN_MISSING, return_value=[FAKE_TILE]) as mock_gen,
            patch(_WRITE_CSV),
        ):
            result = invoke_missing(runner,
                                    ["--short-name", "RWA_23_AEE",
                                     "--outermost_eval_epoch", "ENDLINE"])

        assert result.exit_code == 0, result.output
        _args, kwargs = mock_gen.call_args
        assert kwargs["outermost_eval_epoch"] == "ENDLINE"
        assert kwargs["short_name"] == "RWA_23_AEE"