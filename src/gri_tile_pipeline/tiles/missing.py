"""Generate tiles-CSV rows for polygons missing TTC values.

Reads a TerraMatch geoparquet, finds polygons where ``ttc`` is NULL or empty,
derives one eval year per *distinct* survey-year offset, and spatial-joins
against the tile grid to produce a deduplicated tile list.

Each project phase maps to one or more integer offset years. The offsets are
materialized as an in-SQL ``offsets`` table and cross-joined against the
polygons, so a single query handles every offset at once (no per-offset Python
loop): a polygon is emitted once per offset year for which its ``ttc`` map is
missing that year's value.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from gri_shared_library.constants import (
    TreeCoverProjectPhaseYearRange,
    TCC_BASELINE_OFFSET_YEARS,
    TCC_EI_OFFSET_YEARS,
    TCC_ENDLINE_OFFSET_YEARS,
)
# from gri_shared_library.productivity_tools import debug_expand_number_parameterized_duckdb_query

from gri_tile_pipeline.duckdb_utils import connect_with_spatial


HALF_TILE: float = 1.0 / 36  # half of 1/18 degree tile size


def generate_missing_tiles(
    geoparquet: str,
    tiledb: str,
    outermost_project_phase_name: str = 'BASELINE',
    *,
    short_name: str | None = None,
    framework_key: str | None = None,
    polygon_ids: list[UUID] | None = None,
) -> list[dict[str, Any]]:
    """Spatial join polygons-with-missing-ttc against the tile grid.

    Returns tile dicts deduplicated on (year, X_tile, Y_tile), sorted by year.

    Raises:
        Any exception from the underlying DuckDB query. Failures are *not*
        swallowed: a query error must not be reported to callers as an empty
        (or partial) result, otherwise a genuine failure looks like
        "nothing to do" or a silently truncated output.
    """
    survey_year_offsets = _distinct_offset_years(outermost_project_phase_name)
    current_year = datetime.today().year
    offsets_values = _offsets_values_clause(survey_year_offsets)

    params: list[Any] = [geoparquet]  # $1
    where_clause = _missing_ttc_where(
        "o.off", current_year, short_name, framework_key, polygon_ids, params
    )
    params.append(tiledb)
    tiledb_param = f"${len(params)}"

    query = f"""
        WITH offsets(off) AS (VALUES {offsets_values}),
        polys AS (
            SELECT geom, YEAR(plantstart) + o.off AS yr
            FROM read_parquet($1), offsets o
            WHERE {where_clause}
        )
        SELECT DISTINCT
            p.yr AS year,
            t.X AS lon,
            t.Y AS lat,
            t.X_tile,
            t.Y_tile
        FROM polys p, read_parquet({tiledb_param}) t
        WHERE ST_Intersects(
            p.geom,
            ST_MakeEnvelope(
                t.X - {HALF_TILE}, t.Y - {HALF_TILE},
                t.X + {HALF_TILE}, t.Y + {HALF_TILE}
            )
        )
        ORDER BY p.yr, t.X_tile, t.Y_tile
    """
    # debug_expanded_sql = debug_expand_number_parameterized_duckdb_query(query, params)
    con = connect_with_spatial()
    try:
        all_years_rows = con.execute(query, params).fetchall()
    finally:
        con.close()

    # SELECT DISTINCT already dedupes; keep an explicit guard on the documented
    # (year, X_tile, Y_tile) key in case the projected columns ever change.
    seen: set[tuple[int, int, int]] = set()
    result: list[dict[str, Any]] = []
    for year, lon, lat, x_tile, y_tile in all_years_rows:
        key = (int(year), int(x_tile), int(y_tile))
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "year": int(year),
                "lon": round(float(lon), 4),
                "lat": round(float(lat), 4),
                "X_tile": int(x_tile),
                "Y_tile": int(y_tile),
            }
        )
    return result


def summarize_missing(
    geoparquet: str,
    outermost_project_phase_name: str = 'BASELINE',
) -> dict[str, Any]:
    """Return a structured summary of polygons missing TTC, by cohort and project."""
    survey_year_offsets = _distinct_offset_years(outermost_project_phase_name)
    current_year = datetime.today().year
    offsets_values = _offsets_values_clause(survey_year_offsets)

    params: list[Any] = [geoparquet]  # $1
    where_clause = _missing_ttc_where("o.off", current_year, None, None, None, params)

    # DISTINCT collapses a polygon missing for several offsets to one row,
    # matching the prior UNION-based semantics.
    missing_cte = f"""
        WITH offsets(off) AS (VALUES {offsets_values}),
        missing AS (
            SELECT DISTINCT poly_uuid, framework_key, short_name
            FROM read_parquet($1), offsets o
            WHERE {where_clause}
        )
    """

    con = connect_with_spatial()
    try:
        cohort_rows = con.execute(
            missing_cte
            + "SELECT framework_key, COUNT(*) as cnt FROM missing GROUP BY 1 ORDER BY cnt DESC",
            params,
        ).fetchall()
        project_rows = con.execute(
            missing_cte
            + "SELECT short_name, framework_key, COUNT(*) as cnt FROM missing GROUP BY 1, 2 ORDER BY cnt DESC",
            params,
        ).fetchall()
    finally:
        con.close()

    return {
        "by_cohort": [
            {"framework_key": k, "polygons_missing_ttc": c} for k, c in cohort_rows
        ],
        "by_project": [
            {"short_name": s, "framework_key": k, "polygons_missing_ttc": c}
            for s, k, c in project_rows
        ],
        "total_missing": sum(r[1] for r in cohort_rows),
    }


def list_polygons_missing_ttc(
    geoparquet: str,
    outermost_project_phase_name: str = 'BASELINE',
    *,
    short_name: str | None = None,
    framework_key: str | None = None,
    polygon_ids: list[UUID] | None = None,
    ) -> list[dict[str, Any]]:
    """
    Returns polygon ids for polygons missing ttc for specified outermost project phase.

    Raises:
        Any exception from the underlying DuckDB query. Failures are *not*
        swallowed: see ``generate_missing_tiles``.
    """
    survey_year_offsets = _distinct_offset_years(outermost_project_phase_name)
    current_year = datetime.today().year
    offsets_values = _offsets_values_clause(survey_year_offsets)

    params: list[Any] = [geoparquet]  # $1
    where_clause = _missing_ttc_where(
        "o.off", current_year, short_name, framework_key, polygon_ids, params
    )

    query = f"""
            WITH offsets(off) AS (VALUES {offsets_values})
            SELECT YEAR(plantstart) + o.off AS eval_year, project_id, short_name as project_short_name,
             poly_uuid, plantstart, ST_AsText(geom) as geometry
            FROM read_parquet($1), offsets o
            WHERE {where_clause}
            """
    # debug_expanded_sql = debug_expand_number_parameterized_duckdb_query(query, params)
    con = connect_with_spatial()
    try:
        all_years_rows = con.execute(query, params).fetchall()
    finally:
        con.close()

    return [
        {
            "eval_year": int(eval_year),
            "project_id": project_id,
            "project_short_name": project_short_name,
            "poly_uuid": poly_uuid,
            "plantstart": plantstart,
            "geometry": geometry,
        }
        for eval_year, project_id, project_short_name, poly_uuid, plantstart, geometry in all_years_rows
    ]


def _get_survey_year_offssets(outermost_project_phase_name: str | None) -> list[int]:
    """Return the list of integer offset years that apply to a project phase.

    A phase includes every offset up to and including itself.
    """
    window_offsets = {
        TreeCoverProjectPhaseYearRange.BASELINE.name: [TCC_BASELINE_OFFSET_YEARS],
        TreeCoverProjectPhaseYearRange.EARLY_INSIGHT.name: [
            TCC_BASELINE_OFFSET_YEARS,
            TCC_EI_OFFSET_YEARS,
        ],
        TreeCoverProjectPhaseYearRange.ENDLINE.name: [
            TCC_BASELINE_OFFSET_YEARS,
            TCC_EI_OFFSET_YEARS,
            TCC_ENDLINE_OFFSET_YEARS,
        ],
    }

    # Get list of offset years based on the specified phase name
    key = (outermost_project_phase_name or TreeCoverProjectPhaseYearRange.BASELINE.name).upper()
    if key not in window_offsets:
        raise ValueError(f"Unknown project phase name: {outermost_project_phase_name}")

    return window_offsets[key]


def _distinct_offset_years(outermost_project_phase_name: str | None) -> list[int]:
    """Return a phase's offset years as a sorted list of distinct integers.

    This is what the WHERE clause matches against: distinct integer offsets.
    Offsets shared across phases are collapsed.
    """
    offsets = _get_survey_year_offssets(outermost_project_phase_name)
    return sorted({int(offset) for offset in offsets})


def _offsets_values_clause(offsets: list[int]) -> str:
    """Render offsets as a SQL VALUES body, e.g. ``(-1), (1), (4)``.

    Safe to inline because every element is coerced to ``int`` in
    ``_distinct_offset_years`` (no user-supplied strings reach the SQL text).
    """
    if not offsets:
        raise ValueError("No offset years to query")
    return ", ".join(f"({int(o)})" for o in offsets)


def _missing_ttc_where(
        offset_expr: str,
        current_year: int,
        short_name: str | None,
        framework_key: str | None,
        polygon_ids: list[UUID] | None,
        params: list[Any]) -> str:
    """Build the missing-ttc WHERE clause, appending bind values to ``params``.

    ``offset_expr`` is the SQL expression yielding a row's offset year (e.g.
    ``"o.off"`` when cross-joining an offsets table). Positional parameters are
    numbered from ``len(params) + 1``, so the caller must have already appended
    any params referenced earlier in the query (e.g. the geoparquet path bound
    as ``$1``).
    """
    conditions = [
        f"(ttc IS NULL OR cardinality(ttc) = 0 "
        f"OR NOT list_contains(map_keys(ttc), YEAR(plantstart) + {offset_expr}))"
    ]

    params.append(current_year)
    conditions.append(f"YEAR(plantstart) + {offset_expr} < ${len(params)}")

    conditions.append("ST_IsValid(geom)")

    if short_name is not None:
        params.append(short_name)
        conditions.append(f"short_name = ${len(params)}")
    if framework_key is not None:
        params.append(framework_key)
        conditions.append(f"framework_key = ${len(params)}")
    if polygon_ids is not None:
        # Verify that values are UUIDS
        validated_ids = [str(u if isinstance(u, UUID) else UUID(str(u))) for u in polygon_ids]
        if len(validated_ids) != len(polygon_ids):
            raise ValueError('Some values in polygon_uuids are not valid uuids')
        params.append(validated_ids)
        conditions.append(f"list_contains(${len(params)}, poly_uuid)")

    return " AND ".join(conditions)
