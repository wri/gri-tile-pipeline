"""Generate tiles-CSV rows for polygons missing TTC values.

Reads a TerraMatch geoparquet, finds polygons where ``ttc`` is NULL or empty,
derives ``pred_year = YEAR(plantstart) - 1``, and spatial-joins against the
tile grid to produce a deduplicated tile list.
"""

from __future__ import annotations

from typing import Any

from gri_tile_pipeline.duckdb_utils import connect_with_spatial


HALF_TILE = 1.0 / 36  # half of 1/18 degree tile size


def generate_missing_tiles(
    geoparquet: str,
    tiledb: str,
    *,
    short_name: str | None = None,
    framework_key: str | None = None,
) -> list[dict[str, Any]]:
    """Spatial join polygons-with-missing-ttc against the tile grid.

    Returns tile dicts deduplicated on (year, X_tile, Y_tile), sorted by year.
    """
    con = connect_with_spatial()
    try:
        conditions = ["(ttc IS NULL OR cardinality(ttc) = 0)", "ST_IsValid(geom)"]
        params: list[Any] = [geoparquet]
        param_idx = 2

        if short_name is not None:
            conditions.append(f"short_name = ${param_idx}")
            params.append(short_name)
            param_idx += 1
        if framework_key is not None:
            conditions.append(f"framework_key = ${param_idx}")
            params.append(framework_key)
            param_idx += 1

        where = " AND ".join(conditions)
        params.append(tiledb)
        tiledb_param = f"${param_idx}"

        query = f"""
            WITH polys AS (
                SELECT geom, YEAR(plantstart) - 1 AS yr
                FROM read_parquet($1)
                WHERE {where}
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
        rows = con.execute(query, params).fetchall()
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
        for year, lon, lat, x_tile, y_tile in rows
    ]


def summarize_missing(geoparquet: str) -> dict[str, Any]:
    """Return a structured summary of polygons missing TTC, by cohort and project."""
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
