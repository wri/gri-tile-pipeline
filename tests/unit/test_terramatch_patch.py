"""Tests for the TerraMatch patch orchestration layer."""

from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import MagicMock

from gri_tile_pipeline.terramatch.client import TMApiError
from gri_tile_pipeline.terramatch.patch import (
    IndicatorSpec,
    build_indicator,
    detect_uncertainty_column,
    load_results,
    run_patch,
    summarize,
)


@pytest.fixture
def results_csv(tmp_path):
    path = tmp_path / "results.csv"
    pd.DataFrame([
        {"poly_uuid": "poly-a", "ttc": 80.0, "ttc_shift_error": 2.5, "year": 2023},
        {"poly_uuid": "poly-b", "ttc": 65.0, "ttc_shift_error": 1.1, "year": 2024},
    ]).to_csv(path, index=False)
    return path


# -- detect_uncertainty_column ------------------------------------------------

def test_detect_uncertainty_picks_primary_candidate():
    cols = ["poly_uuid", "ttc", "ttc_shift_error", "year"]
    assert detect_uncertainty_column(cols) == "ttc_shift_error"


def test_detect_uncertainty_suffix_fallback():
    cols = ["poly_uuid", "ttc", "my_custom_error"]
    assert detect_uncertainty_column(cols) == "my_custom_error"


def test_detect_uncertainty_none_when_absent():
    assert detect_uncertainty_column(["poly_uuid", "ttc", "year"]) is None


# -- build_indicator ----------------------------------------------------------

def test_build_indicator_shape_and_values():
    spec = IndicatorSpec(
        year=2020, slug="treeCover", project_phase="implementation",
        year_column="year", percent_column="ttc",
        uncertainty_column="ttc_shift_error",
    )
    row = {"poly_uuid": "poly-a", "ttc": 80.0, "ttc_shift_error": 2.5, "year": 2023}
    ind = build_indicator(row, spec)
    assert ind == {
        "indicatorSlug": "treeCover",
        "yearOfAnalysis": 2023,  # row override wins
        "projectPhase": "implementation",
        "percentCover": 80.0,
        "plusMinusPercent": 2.5,
    }


def test_build_indicator_uses_spec_year_when_row_missing():
    spec = IndicatorSpec(year=2021, year_column="pred_year")
    row = {"poly_uuid": "poly-a", "ttc": 50.0}
    ind = build_indicator(row, spec)
    assert ind["yearOfAnalysis"] == 2021


def test_build_indicator_requires_year_somewhere():
    spec = IndicatorSpec(year=None, year_column="year")
    row = {"poly_uuid": "poly-a", "ttc": 50.0}
    with pytest.raises(ValueError, match="no year"):
        build_indicator(row, spec)


def test_build_indicator_requires_numeric_percent():
    spec = IndicatorSpec(year=2023)
    row = {"poly_uuid": "poly-a", "ttc": None}
    with pytest.raises(ValueError, match="ttc"):
        build_indicator(row, spec)


def test_build_indicator_omits_uncertainty_when_nan():
    import math
    spec = IndicatorSpec(
        year=2023, year_column="year",
        uncertainty_column="ttc_shift_error",
    )
    row = {"poly_uuid": "poly-a", "ttc": 50.0, "ttc_shift_error": math.nan, "year": 2023}
    ind = build_indicator(row, spec)
    assert "plusMinusPercent" not in ind


# -- run_patch ----------------------------------------------------------------

def _client_with(known_ids: list[str], patch_responses: list | None = None):
    client = MagicMock()
    client.list_site_polygons.return_value = iter(
        [{"id": pid} for pid in known_ids]
    )
    if patch_responses is None:
        ok = MagicMock()
        ok.status_code = 200
        client.patch_site_polygon.return_value = ok
    else:
        client.patch_site_polygon.side_effect = patch_responses
    return client


def test_run_patch_dry_run_never_calls_patch(results_csv):
    rows, cols = load_results(results_csv)
    spec = IndicatorSpec(year=None, year_column="year",
                         uncertainty_column="ttc_shift_error")
    client = _client_with(["poly-a", "poly-b"])
    outcomes = run_patch(rows, "proj-1", client, spec, apply=False)
    assert [o.status for o in outcomes] == ["dryrun", "dryrun"]
    client.patch_site_polygon.assert_not_called()


def test_run_patch_apply_sends_patches(results_csv):
    rows, cols = load_results(results_csv)
    spec = IndicatorSpec(year=None, year_column="year",
                         uncertainty_column="ttc_shift_error")
    client = _client_with(["poly-a", "poly-b"])
    outcomes = run_patch(rows, "proj-1", client, spec, apply=True)
    assert [o.status for o in outcomes] == ["sent", "sent"]
    assert client.patch_site_polygon.call_count == 2
    args, _ = client.patch_site_polygon.call_args_list[0]
    assert args[0] == "poly-a"
    assert args[1][0]["indicatorSlug"] == "treeCover"
    assert args[1][0]["yearOfAnalysis"] == 2023


def test_run_patch_unmatched_row_reported(results_csv):
    rows, _ = load_results(results_csv)
    spec = IndicatorSpec(year=None, year_column="year",
                         uncertainty_column="ttc_shift_error")
    # Only poly-a is known; poly-b should be reported unmatched.
    client = _client_with(["poly-a"])
    outcomes = run_patch(rows, "proj-1", client, spec, apply=True)
    status_by_uuid = {o.poly_uuid: o.status for o in outcomes}
    assert status_by_uuid == {"poly-a": "sent", "poly-b": "unmatched"}
    client.patch_site_polygon.assert_called_once()


def test_run_patch_continues_after_api_error(results_csv):
    rows, _ = load_results(results_csv)
    spec = IndicatorSpec(year=None, year_column="year",
                         uncertainty_column="ttc_shift_error")
    ok = MagicMock(); ok.status_code = 200
    client = _client_with(
        ["poly-a", "poly-b"],
        patch_responses=[TMApiError(500, "boom"), ok],
    )
    outcomes = run_patch(rows, "proj-1", client, spec, apply=True)
    assert [o.status for o in outcomes] == ["error", "sent"]
    assert outcomes[0].http_status == 500
    assert "boom" in (outcomes[0].message or "")


def test_run_patch_limit_applies(results_csv):
    rows, _ = load_results(results_csv)
    spec = IndicatorSpec(year=None, year_column="year")
    client = _client_with(["poly-a", "poly-b"])
    outcomes = run_patch(rows, "proj-1", client, spec, apply=True, limit=1)
    assert len(outcomes) == 1
    assert outcomes[0].poly_uuid == "poly-a"


def test_run_patch_row_missing_percent_is_error(tmp_path):
    path = tmp_path / "broken.csv"
    pd.DataFrame([{"poly_uuid": "poly-a", "ttc": None, "year": 2023}]).to_csv(path, index=False)
    rows, _ = load_results(path)
    spec = IndicatorSpec(year=None, year_column="year")
    client = _client_with(["poly-a"])
    outcomes = run_patch(rows, "proj-1", client, spec, apply=True)
    assert outcomes[0].status == "error"
    client.patch_site_polygon.assert_not_called()


def test_run_patch_missing_poly_uuid_column(tmp_path):
    path = tmp_path / "no_uuid.csv"
    pd.DataFrame([{"ttc": 50.0, "year": 2023}]).to_csv(path, index=False)
    rows, _ = load_results(path)
    spec = IndicatorSpec(year=None, year_column="year")
    client = _client_with(["poly-a"])
    outcomes = run_patch(rows, "proj-1", client, spec, apply=True)
    assert outcomes[0].status == "error"


# -- summarize ----------------------------------------------------------------

def test_summarize_counts():
    from gri_tile_pipeline.terramatch.patch import PatchOutcome
    outs = [
        PatchOutcome(poly_uuid="a", polygon_id="a", status="sent"),
        PatchOutcome(poly_uuid="b", polygon_id="b", status="sent"),
        PatchOutcome(poly_uuid="c", polygon_id=None, status="unmatched"),
        PatchOutcome(poly_uuid="d", polygon_id="d", status="error"),
    ]
    s = summarize(outs)
    assert s == {"sent": 2, "dryrun": 0, "unmatched": 1, "error": 1, "total": 4}
