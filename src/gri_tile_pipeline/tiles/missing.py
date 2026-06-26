"""Generate tiles-CSV rows for polygons missing TTC values.

Reads a TerraMatch geoparquet, finds polygons where ``ttc`` is NULL or empty,
derives ``pred_year = YEAR(plantstart) - 1``, and spatial-joins against the
tile grid to produce a deduplicated tile list.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Tuple
from uuid import UUID

from gri_shared_library.constants import TreeCoverProjectPhaseYearRange

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

    # Loop through offset years, identifying missing tiles
    all_years_rows = []
    con = connect_with_spatial()
    try:
        for offset_year in survey_year_offsets:
            where_clause, params, param_idx = (
                _construct_where_clause_for_geoparquet(geoparquet, short_name, framework_key, offset_year, polygon_ids))

            params.append(tiledb)
            tiledb_param = f"${param_idx}"

            query = f"""
                WITH polys AS (
                    SELECT geom, YEAR(plantstart) + $2 AS yr
                    FROM read_parquet($1)
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
            this_year_rows = con.execute(query, params).fetchall()
            all_years_rows.extend(this_year_rows)
    finally:
        con.close()

    return [
        {
            "year": int(year),
            "lon": round(float(lon), 4),
            "lat": round(float(lat), 4),
            "X_tile": int(x_tile),
            "Y_tile": int(y_tile),
        }
        for year, lon, lat, x_tile, y_tile in all_years_rows
    ]


def summarize_missing(
    geoparquet: str,
    outermost_project_phase_name: str = 'BASELINE',
) -> dict[str, Any]:
    """Return a structured summary of polygons missing TTC, by cohort and project.

    Args:
        geoparquet: geoparquet fileoath
        outermost_project_phase_name: name of the outermost project phase
    """
    import re
    survey_year_offsets = _get_survey_year_offssets(outermost_project_phase_name)

    subqueries: list[str] = []
    all_params: list[Any] = []

    for offset_year in survey_year_offsets:
        where_clause, params, _ = _construct_where_clause_for_geoparquet(
            geoparquet, None, None, offset_year, None
        )
        shift = len(all_params)
        shifted_where = re.sub(r'\$(\d+)', lambda m: f'${int(m.group(1)) + shift}', where_clause)
        subqueries.append(
            f"SELECT poly_uuid, framework_key, short_name "
            f"FROM read_parquet(${1 + shift}) WHERE {shifted_where}"
        )
        all_params.extend(params)

    union_sql = " UNION ".join(subqueries)

    con = connect_with_spatial()
    try:
        cohort_rows = con.execute(
            f"""
            SELECT framework_key, COUNT(*) as cnt
            FROM ({union_sql})
            GROUP BY 1 ORDER BY cnt DESC
            """,
            all_params,
        ).fetchall()
        project_rows = con.execute(
            f"""
            SELECT short_name, framework_key, COUNT(*) as cnt
            FROM ({union_sql})
            GROUP BY 1, 2
            ORDER BY cnt DESC
            """,
            all_params,
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

    # Loop through offset years, identifying polygons missing ttc for project phase year
    all_years_rows = []
    con = connect_with_spatial()
    try:
        for offset_year in survey_year_offsets:
            where_clause, params, param_idx = (
                _construct_where_clause_for_geoparquet(geoparquet, short_name, framework_key, offset_year, polygon_ids))

            query = f"""
                    SELECT YEAR(plantstart) + $2 AS eval_year, project_id, short_name as project_short_name,
                     poly_uuid, plantstart, ST_AsText(geom) as geometry
                    FROM read_parquet($1)
                    WHERE {where_clause}
                    """
            # debug_expanded_sql = debug_expand_number_parameterized_duckdb_query(query, params)
            this_year_rows = con.execute(query, params).fetchall()
            all_years_rows.extend(this_year_rows)
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


def _get_survey_year_offssets(outermost_project_phase_name):
    epoch_offsets = {
        TreeCoverProjectPhaseYearRange.BASELINE.name: [TreeCoverProjectPhaseYearRange.BASELINE.value],
        TreeCoverProjectPhaseYearRange.EARLY_INSIGHTS.name: [TreeCoverProjectPhaseYearRange.BASELINE.value, TreeCoverProjectPhaseYearRange.EARLY_INSIGHTS.value],
        TreeCoverProjectPhaseYearRange.ENDLINE.name: [TreeCoverProjectPhaseYearRange.BASELINE.value, TreeCoverProjectPhaseYearRange.EARLY_INSIGHTS.value, TreeCoverProjectPhaseYearRange.ENDLINE.value],
    }

    # Get list of offset years based on specified outermost_project_phase_name
    key = (outermost_project_phase_name or TreeCoverProjectPhaseYearRange.BASELINE.name).upper()
    if key not in epoch_offsets:
        raise ValueError(f"Unknown project phase name: {outermost_project_phase_name}")
    survey_year_offsets = epoch_offsets[key]

    return survey_year_offsets


def _construct_where_clause_for_geoparquet(
        geoparquet: str,
        short_name: str,
        framework_key: str,
        offset_year_range: Tuple[int, int],
        polygon_ids: list[UUID] | None = None):
    """
    Constructs WHERE clause for querying geoparquet file with qualification for TTC, plantstart, and optional filters.
    """
    offset_year_start = offset_year_range[0]
    offset_year_end = offset_year_range[1]

    current_year = datetime.today().year
    conditions = ["(ttc IS NULL OR cardinality(ttc) = 0 OR "
                  "len(list_filter(map_keys(ttc), k -> k BETWEEN YEAR(plantstart) + $2 AND YEAR(plantstart) + $3)) = 0)",
                  f"YEAR(plantstart) + $3 < $4",
                  "ST_IsValid(geom)"]
    params: list[Any] = [geoparquet, offset_year_start, offset_year_end, current_year]
    param_idx = 5

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
