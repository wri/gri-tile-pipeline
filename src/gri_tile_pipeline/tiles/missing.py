"""Generate tiles-CSV rows for polygons missing TTC values.

Reads a TerraMatch geoparquet, finds polygons where ``ttc`` is NULL or empty,
derives ``pred_year = YEAR(plantstart) - 1``, and spatial-joins against the
tile grid to produce a deduplicated tile list.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from gri_shared_library.constants import TreeCoverProjectPhaseYearRange
from gri_shared_library.productivity_tools import debug_expand_number_parameterized_duckdb_query

from gri_tile_pipeline.duckdb_utils import connect_with_spatial


HALF_TILE = 1.0 / 36  # half of 1/18 degree tile size


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
    NOTE: The code only identifies tiles without TTC (tree_cover) value in the tm.geoparquet file. It does not
    query TerraMatch API directly.

    Args:
        geoparquet: geoparquet fileoath
        tiledb: tiledb filepath
        outermost_project_phase_name: name of the outermost project phase
        short_name: optional project_short_name for filtering TM records
        framework_key: optional framework_key for filtering TM records
        polygon_ids: optional list of polygons ids for filtering TM records

    Returns: tile dicts deduplicated on (year, X_tile, Y_tile), sorted by year.

    Raises:
        Any exception from the underlying DuckDB query. Failures are *not*
        swallowed: a query error must not be reported to callers as an empty
        (or partial) result, otherwise a genuine failure is a silently truncated output.
    """
    survey_year_offsets = _get_survey_year_offssets(outermost_project_phase_name)
    offsets = _distinct_year_offsets(survey_year_offsets)
    current_year = datetime.today().year

    # Single geoparquet scan cross-joined against the set of distinct year
    # offsets, so the evaluation year is the per-row YEAR(plantstart) + o.off.
    # DISTINCT collapses tiles shared across overlapping year ranges.
    where_clause, params, param_idx = _construct_query_filters(
        geoparquet, current_year, offsets, short_name, framework_key, polygon_ids)

    params.append(tiledb)
    tiledb_param = f"${param_idx}"

    query = f"""
        WITH polys AS (
            SELECT geom, YEAR(plantstart) + o.off AS yr
            FROM read_parquet($1), (SELECT UNNEST($3::INTEGER[]) AS off) o
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

    con = connect_with_spatial()
    try:
        # debug_expanded_sql = debug_expand_number_parameterized_duckdb_query(query, params)
        all_years_rows = con.execute(query, params).fetchall()
    finally:
        con.close()

    results = [
        {
            "year": int(year),
            "lon": round(float(lon), 4),
            "lat": round(float(lat), 4),
            "X_tile": int(x_tile),
            "Y_tile": int(y_tile),
        }
        for year, lon, lat, x_tile, y_tile in all_years_rows
    ]
    return results


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

    Args:
        geoparquet: geoparquet fileoath
        outermost_project_phase_name: name of the outermost project phase
        short_name: optional project_short_name for filtering TM records
        framework_key: optional framework_key for filtering TM records
        polygon_ids: optional list of polygons ids for filtering TM records

    Raises:
        Any exception from the underlying DuckDB query. Failures are *not*
        swallowed: see ``generate_missing_tiles``.
    """
    survey_year_offsets = _get_survey_year_offssets(outermost_project_phase_name)
    offsets = _distinct_year_offsets(survey_year_offsets)
    current_year = datetime.today().year

    # Single scan cross-joined against the distinct offsets. DISTINCT removes
    # (polygon, year) rows duplicated by overlapping year ranges.
    where_clause, params, _ = _construct_query_filters(
        geoparquet, current_year, offsets, short_name, framework_key, polygon_ids)

    query = f"""
        SELECT DISTINCT
            YEAR(plantstart) + o.off AS eval_year,
            project_id,
            short_name AS project_short_name,
            poly_uuid,
            plantstart,
            ST_AsText(geom) AS geometry
        FROM read_parquet($1), (SELECT UNNEST($3::INTEGER[]) AS off) o
        WHERE {where_clause}
        ORDER BY eval_year, poly_uuid
    """

    con = connect_with_spatial()
    try:
        # debug_expanded_sql = debug_expand_number_parameterized_duckdb_query(query, params)
        all_years_rows = con.execute(query, params).fetchall()
    finally:
        con.close()

    results = [
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

    return results


def summarize_missing(
    geoparquet: str,
    outermost_project_phase_name: str = 'BASELINE',
) -> dict[str, Any]:
    """Return a structured summary of polygons missing TTC, by cohort and project.

    Args:
        geoparquet: geoparquet fileoath
        outermost_project_phase_name: name of the outermost project phase
    """
    survey_year_offsets = _get_survey_year_offssets(outermost_project_phase_name)
    offsets = _distinct_year_offsets(survey_year_offsets)
    current_year = datetime.today().year

    where_clause, params, _ = _construct_query_filters(
        geoparquet, current_year, offsets, None, None, None)

    con = connect_with_spatial()
    try:
        # Materialize the distinct set of missing polygons once, then run all
        # three rollups against the temp table so the geoparquet is scanned a
        # single time. COUNT(DISTINCT poly_uuid) guards the counts regardless.
        con.execute(
            f"""
            CREATE TEMP TABLE missing_polys AS
            SELECT DISTINCT poly_uuid, framework_key, short_name
            FROM read_parquet($1), (SELECT UNNEST($3::INTEGER[]) AS off) o
            WHERE {where_clause}
            """,
            params,
        )
        cohort_rows = con.execute(
            """
            SELECT framework_key, COUNT(DISTINCT poly_uuid) AS cnt
            FROM missing_polys
            GROUP BY 1 ORDER BY cnt DESC
            """
        ).fetchall()
        project_rows = con.execute(
            """
            SELECT short_name, framework_key, COUNT(DISTINCT poly_uuid) AS cnt
            FROM missing_polys
            GROUP BY 1, 2 ORDER BY cnt DESC
            """
        ).fetchall()
        total_missing = con.execute(
            "SELECT COUNT(DISTINCT poly_uuid) FROM missing_polys"
        ).fetchone()[0]
    finally:
        con.close()

    results = {
        "by_cohort": [
            {"framework_key": k, "polygons_missing_ttc": c} for k, c in cohort_rows
        ],
        "by_project": [
            {"short_name": s, "framework_key": k, "polygons_missing_ttc": c}
            for s, k, c in project_rows
        ],
        "total_missing": total_missing,
    }

    return results


def _get_survey_year_offssets(outermost_project_phase_name):
    window_offsets = {
        TreeCoverProjectPhaseYearRange.BASELINE.name: [TreeCoverProjectPhaseYearRange.BASELINE.value],
        TreeCoverProjectPhaseYearRange.EARLY_INSIGHT.name: [TreeCoverProjectPhaseYearRange.BASELINE.value,
                                                             TreeCoverProjectPhaseYearRange.EARLY_INSIGHT.value],
        TreeCoverProjectPhaseYearRange.ENDLINE.name: [TreeCoverProjectPhaseYearRange.BASELINE.value,
                                                      TreeCoverProjectPhaseYearRange.EARLY_INSIGHT.value,
                                                      TreeCoverProjectPhaseYearRange.ENDLINE.value],
    }

    # Get list of offset (start, end) ranges based on specified outermost_project_phase_name
    key = (outermost_project_phase_name or TreeCoverProjectPhaseYearRange.BASELINE.name).upper()
    if key not in window_offsets:
        raise ValueError(f"Unknown project phase name: {outermost_project_phase_name}")
    survey_year_offsets = window_offsets[key]

    return survey_year_offsets


def _distinct_year_offsets(survey_year_offsets) -> list[int]:
    """Flatten a list of inclusive (start, end) offset ranges into the sorted set
    of distinct individual year offsets. Overlapping ranges collapse here, so the
    downstream cross-join never produces duplicate (polygon, year) rows."""
    return sorted({off for (start, end) in survey_year_offsets for off in range(start, end + 1)})


def _construct_query_filters(
        geoparquet: str,
        current_year: int,
        offsets: list[int],
        short_name: str | None,
        framework_key: str | None,
        polygon_ids: list[UUID] | None = None):
    """
    Build the shared WHERE clause and positional params for the geoparquet scan.

    The scan is cross-joined against the distinct year offsets, so the evaluation
    year is the per-row ``YEAR(plantstart) + o.off`` (``o.off`` comes from
    ``UNNEST($3::INTEGER[])`` in the caller's FROM clause). Fixed param layout:

        $1   = geoparquet path
        $2   = current_year
        $3   = offsets (INTEGER[])   -- consumed by UNNEST in the FROM clause
        $4.. = optional short_name / framework_key / polygon_ids filters

    Returns (where_clause, params, param_idx) where ``param_idx`` is the next free
    positional index, for callers that append further params (e.g. tiledb).
    """
    conditions = ["(ttc IS NULL OR cardinality(ttc) = 0 OR NOT list_contains(map_keys(ttc), YEAR(plantstart) + o.off))",
                  "YEAR(plantstart) + o.off < $2",
                  "ST_IsValid(geom)"]
    params: list[Any] = [geoparquet, current_year, offsets]
    param_idx = 4

    if short_name is not None:
        conditions.append(f"short_name = ${param_idx}")
        params.append(short_name)
        param_idx += 1
    if framework_key is not None:
        conditions.append(f"framework_key = ${param_idx}")
        params.append(framework_key)
        param_idx += 1
    if polygon_ids is not None:
        # Verify that values are UUIDS
        validated_ids = [str(u if isinstance(u, UUID) else UUID(str(u))) for u in polygon_ids]
        if len(validated_ids) != len(polygon_ids):
            raise ValueError('Some values in polygon_uuids are not valid uuids')

        conditions.append(f"list_contains(${param_idx}, poly_uuid)")
        params.append(validated_ids)
        param_idx += 1

    where_clause = " AND ".join(conditions)
    return where_clause, params, param_idx