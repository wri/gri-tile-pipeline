"""Generate tiles-CSV rows for polygons missing TTC values.

Reads a TerraMatch geoparquet, finds polygons where ``ttc`` is NULL or empty,
derives ``pred_year = YEAR(plantstart) - 1``, and spatial-joins against the
tile grid to produce a deduplicated tile list.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import pandas as pd
from gri_shared_library.constants import EvalEpoch
from gri_tile_pipeline.duckdb_utils import connect_with_spatial


HALF_TILE = 1.0 / 36  # half of 1/18 degree tile size


def generate_missing_tiles(
    geoparquet: str,
    tiledb: str,
    outermost_eval_epoch_name: str = 'BASELINE',
    *,
    short_name: str | None = None,
    framework_key: str | None = None,
    polygon_ids: list[UUID] | None = None,
) -> list[dict[str, Any]]:
    """Spatial join polygons-with-missing-ttc against the tile grid.

    Returns tile dicts deduplicated on (year, X_tile, Y_tile), sorted by year.
    """
    survey_year_offsets = _get_survey_year_offssets(outermost_eval_epoch_name)

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
            # debug_expanded_sql = _expand_query(query, params)
            this_year_rows = con.execute(query, params).fetchall()
            all_years_rows.extend(this_year_rows)
    except Exception as e:
        print(f"Error on querying for missing tiles: {e}")  # or log/raise
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


def summarize_missing(geoparquet: str) -> dict[str, Any]:
    """Return a structured summary of polygons missing TTC, by cohort and project."""
    # TODO: Update to summarize by eval epoch
    con = connect_with_spatial()
    try:
        cohort_rows = con.execute(
            """
            SELECT framework_key, COUNT(*) as cnt
            FROM read_parquet($1)
            WHERE ttc IS NULL OR cardinality(ttc) = 0
            GROUP BY 1 ORDER BY cnt DESC
            """,
            [geoparquet],
        ).fetchall()
        project_rows = con.execute(
            """
            SELECT short_name, framework_key, COUNT(*) as cnt
            FROM read_parquet($1)
            WHERE ttc IS NULL OR cardinality(ttc) = 0
            GROUP BY 1, 2 ORDER BY cnt DESC
            """,
            [geoparquet],
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
    outermost_eval_epoch_name: str = 'BASELINE',
    *,
    short_name: str | None = None,
    framework_key: str | None = None,
    polygon_ids: list[UUID] | None = None,
    ) -> list[dict[str, Any]]:
    """
    Returns polygon ids for polygons missing ttc for specified outermost eval epoch.
    """
    survey_year_offsets = _get_survey_year_offssets(outermost_eval_epoch_name)

    # Loop through offset years, identifying polygons missing ttc for eval epoch year
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
            debug_expanded_sql = _expand_query(query, params)
            this_year_rows = con.execute(query, params).fetchall()
            all_years_rows.extend(this_year_rows)
    except Exception as e:
        print(f"Error on querying for missing polygon ttc: {e}")  # or log/raise
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


def _get_survey_year_offssets(outermost_eval_epoch_name):
    epoch_offsets = {
        EvalEpoch.BASELINE.name: [EvalEpoch.BASELINE.value],
        EvalEpoch.MIDWAY.name: [EvalEpoch.BASELINE.value, EvalEpoch.MIDWAY.value],
        EvalEpoch.ENDLINE.name: [EvalEpoch.BASELINE.value, EvalEpoch.MIDWAY.value, EvalEpoch.ENDLINE.value],
    }

    # Get list of offset years based on specified outermost_eval_epoch_name
    key = (outermost_eval_epoch_name or EvalEpoch.BASELINE.name).upper()
    if key not in epoch_offsets:
        raise ValueError(f"Unknown eval epoch name: {outermost_eval_epoch_name}")
    survey_year_offsets = epoch_offsets[key]

    return survey_year_offsets

def _construct_where_clause_for_geoparquet(
        geoparquet: str,
        short_name: str,
        framework_key: str,
        offset_year: int,
        polygon_ids: list[UUID] | None = None):

    current_year = datetime.today().year
    conditions = ["(ttc IS NULL OR cardinality(ttc) = 0 OR NOT list_contains(map_keys(ttc), YEAR(plantstart) + $2))",
                  f"YEAR(plantstart) + $2 < $3",
                  "ST_IsValid(geom)"]
    params: list[Any] = [geoparquet, offset_year, current_year]
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


def _expand_query(query, params):
    for i, p in enumerate(params, 1):
        value = f"'{p}'" if isinstance(p, str) else str(p)
        query = query.replace(f'${i}', value)
    return query
