"""Tests for the ``tiles missing`` CLI command and its project_phase handling.

Covers:
  - TreeCoverProjectPhaseYearRange (start, end) ranges, contains(), from_score()
  - CLI routes project_phase correctly to generate_missing_tiles()
  - All three epoch names (BASELINE / EARLY_INSIGHTS / ENDLINE) map to the right offsets
  - Default epoch is BASELINE
  - Invalid epoch name raises ValueError inside generate_missing_tiles
  - NO_WORK exit path when generate_missing_tiles returns an empty list
  - Summary-only path (no --short-name / --framework-key / --output)
  - Output written when --output is supplied
  - framework_key filter forwarded correctly
  - Combined project_phase + framework_key forwarding
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from gri_tile_pipeline.cli import gri_ttc
from gri_shared_library.constants import TreeCoverProjectPhaseYearRange
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
# Unit tests: TreeCoverProjectPhaseYearRange enum
# ---------------------------------------------------------------------------

class TestTreeCoverProjectPhaseYearRange:
    """Each member exposes a (start, end) year range plus contains()/from_score()."""

    # -- (start, end) values -------------------------------------------------

    def test_baseline_range(self):
        assert TreeCoverProjectPhaseYearRange.BASELINE.start == -2
        assert TreeCoverProjectPhaseYearRange.BASELINE.end == -1

    def test_early_insights_range(self):
        assert TreeCoverProjectPhaseYearRange.EARLY_INSIGHTS.start == 0
        assert TreeCoverProjectPhaseYearRange.EARLY_INSIGHTS.end == 2

    def test_endline_range(self):
        assert TreeCoverProjectPhaseYearRange.ENDLINE.start == 3
        assert TreeCoverProjectPhaseYearRange.ENDLINE.end == 6

    def test_ranges_are_contiguous_and_ordered(self):
        members = list(TreeCoverProjectPhaseYearRange)
        for prev, nxt in zip(members, members[1:]):
            assert prev.end + 1 == nxt.start

    # -- contains() ----------------------------------------------------------

    def test_contains_includes_both_endpoints(self):
        phase = TreeCoverProjectPhaseYearRange.EARLY_INSIGHTS
        assert phase.contains(phase.start)
        assert phase.contains(phase.end)

    def test_contains_inside_range(self):
        assert TreeCoverProjectPhaseYearRange.ENDLINE.contains(4)

    def test_contains_rejects_out_of_range(self):
        assert not TreeCoverProjectPhaseYearRange.BASELINE.contains(0)
        assert not TreeCoverProjectPhaseYearRange.ENDLINE.contains(7)

    # -- from_score() --------------------------------------------------------

    @pytest.mark.parametrize(
        "score, expected",
        [
            (-2, TreeCoverProjectPhaseYearRange.BASELINE),
            (-1, TreeCoverProjectPhaseYearRange.BASELINE),
            (0, TreeCoverProjectPhaseYearRange.EARLY_INSIGHTS),
            (2, TreeCoverProjectPhaseYearRange.EARLY_INSIGHTS),
            (3, TreeCoverProjectPhaseYearRange.ENDLINE),
            (6, TreeCoverProjectPhaseYearRange.ENDLINE),
        ],
    )
    def test_from_score_maps_to_correct_phase(self, score, expected):
        assert TreeCoverProjectPhaseYearRange.from_score(score) is expected

    def test_from_score_uses_end_attribute_for_offsets(self):
        """The CLI/SQL offset ($2) is the phase's ``end`` value."""
        assert TreeCoverProjectPhaseYearRange.from_score(-1).end == -1
        assert TreeCoverProjectPhaseYearRange.from_score(2).end == 2
        assert TreeCoverProjectPhaseYearRange.from_score(6).end == 6

    @pytest.mark.parametrize("score", [-3, 7, 100])
    def test_from_score_raises_when_no_phase_matches(self, score):
        with pytest.raises(StopIteration):
            TreeCoverProjectPhaseYearRange.from_score(score)


# ---------------------------------------------------------------------------
# Unit tests: generate_missing_tiles — project_phase → survey_year_offset mapping
# ---------------------------------------------------------------------------

class TestOutermostTreeCoverProjectPhaseYearRangeOffsetMapping:
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
                outermost_project_phase_name=epoch,
            )

        # params = second positional arg to con.execute(query, params)
        params: list = mock_con.execute.call_args[0][1]
        return params[2]  # index 1 == $2 == survey_year_offset

    def test_baseline_offset_is_minus_one(self):
        assert self._captured_offset("BASELINE") == TreeCoverProjectPhaseYearRange.BASELINE.end  # -1

    def test_midway_offset_is_two(self):
        assert self._captured_offset("EARLY_INSIGHTS") == TreeCoverProjectPhaseYearRange.EARLY_INSIGHTS.end      # 2

    def test_endline_offset_is_six(self):
        assert self._captured_offset("ENDLINE") == TreeCoverProjectPhaseYearRange.ENDLINE.end    # 6

    def test_lowercase_baseline(self):
        assert self._captured_offset("baseline") == TreeCoverProjectPhaseYearRange.BASELINE.end

    def test_lowercase_midway(self):
        assert self._captured_offset("early_insights") == TreeCoverProjectPhaseYearRange.EARLY_INSIGHTS.end

    def test_none_defaults_to_baseline(self):
        """project_phase=None is treated as BASELINE per the implementation guard."""
        assert self._captured_offset(None) == TreeCoverProjectPhaseYearRange.BASELINE.end

    def test_invalid_outermost_epoch_raises_value_error(self):
        mock_con = MagicMock()
        with patch("gri_tile_pipeline.tiles.missing.connect_with_spatial",
                   return_value=mock_con):
            with pytest.raises(ValueError, match="Unknown project phase"):
                generate_missing_tiles(
                    geoparquet="fake.parquet",
                    tiledb="fake_tiledb.parquet",
                    outermost_project_phase_name="QUARTERLY",
                )


# ---------------------------------------------------------------------------
# CLI tests: ``gri-ttc tiles missing``
# ---------------------------------------------------------------------------

class TestTilesMissingCLI:

    # -- project_phase forwarding -----------------------------------------------

    def test_default_outermost_project_phase_name_is_baseline(self, runner):
        with (
            patch(_GEN_MISSING, return_value=[FAKE_TILE]) as mock_gen,
            patch(_WRITE_CSV),
        ):
            result = invoke_missing(runner, ["--short-name", "TEST_23_XXX"])

        assert result.exit_code == 0, result.output
        _args, kwargs = mock_gen.call_args
        assert kwargs.get("outermost_project_phase_name", "BASELINE") == "BASELINE"

    def test_outermost_project_phase_name_baseline_forwarded(self, runner):
        with (
            patch(_GEN_MISSING, return_value=[FAKE_TILE]) as mock_gen,
            patch(_WRITE_CSV),
        ):
            result = invoke_missing(runner,
                                    ["--short-name", "TEST_23_XXX",
                                     "--outermost_project_phase_name", "BASELINE"])

        assert result.exit_code == 0, result.output
        _args, kwargs = mock_gen.call_args
        assert kwargs["outermost_project_phase_name"] == "BASELINE"

    def test_outermost_project_phase_name_midway_forwarded(self, runner):
        with (
            patch(_GEN_MISSING, return_value=[FAKE_TILE]) as mock_gen,
            patch(_WRITE_CSV),
        ):
            result = invoke_missing(runner,
                                    ["--short-name", "TEST_23_XXX",
                                     "--outermost_project_phase_name", "EARLY_INSIGHTS"])

        assert result.exit_code == 0, result.output
        _args, kwargs = mock_gen.call_args
        assert kwargs["outermost_project_phase_name"] == "EARLY_INSIGHTS"

    def test_outermost_project_phase_name_endline_forwarded(self, runner):
        with (
            patch(_GEN_MISSING, return_value=[FAKE_TILE]) as mock_gen,
            patch(_WRITE_CSV),
        ):
            result = invoke_missing(runner,
                                    ["--short-name", "TEST_23_XXX",
                                     "--outermost_project_phase_name", "ENDLINE"])

        assert result.exit_code == 0, result.output
        _args, kwargs = mock_gen.call_args
        assert kwargs["outermost_project_phase_name"] == "ENDLINE"

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

    # -- combined project_phase + framework_key ---------------------------------

    def test_midway_with_framework_key(self, runner):
        with (
            patch(_GEN_MISSING, return_value=[FAKE_TILE]) as mock_gen,
            patch(_WRITE_CSV),
        ):
            result = invoke_missing(runner,
                                    ["--framework-key", "hbf",
                                     "--outermost_project_phase_name", "EARLY_INSIGHTS"])

        assert result.exit_code == 0, result.output
        _args, kwargs = mock_gen.call_args
        assert kwargs["outermost_project_phase_name"] == "EARLY_INSIGHTS"
        assert kwargs["framework_key"] == "hbf"

    def test_endline_with_short_name(self, runner):
        with (
            patch(_GEN_MISSING, return_value=[FAKE_TILE]) as mock_gen,
            patch(_WRITE_CSV),
        ):
            result = invoke_missing(runner,
                                    ["--short-name", "RWA_23_AEE",
                                     "--outermost_project_phase_name", "ENDLINE"])

        assert result.exit_code == 0, result.output
        _args, kwargs = mock_gen.call_args
        assert kwargs["outermost_project_phase_name"] == "ENDLINE"
        assert kwargs["short_name"] == "RWA_23_AEE"