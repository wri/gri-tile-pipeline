#!/usr/bin/env python
"""Generate a CSV of tiles needed for polygons missing TTC values.

Reads a TerraMatch geoparquet, finds polygons where ``ttc`` is NULL,
computes the target prediction year as ``YEAR(plantstart) - 1``, and
performs a spatial join against the tile grid to produce a deduplicated
tile list sorted by year.

Usage:
    # List projects/cohorts with missing ttc counts:
    python scripts/generate_missing_ttc_tiles.py

    # Generate CSV for all missing-ttc polygons:
    python scripts/generate_missing_ttc_tiles.py --output missing_ttc_tiles.csv

    # Filter by project:
    python scripts/generate_missing_ttc_tiles.py \
        --short-name RWA_23_AEE --output missing_rwa_aee.csv

    # Filter by cohort:
    python scripts/generate_missing_ttc_tiles.py \
        --framework-key hbf --output missing_hbf.csv

    # Combined filter:
    python scripts/generate_missing_ttc_tiles.py \
        --framework-key terrafund-landscapes --short-name RWA_23_AEE \
        --output missing.csv

    # Full stats (existing + missing TTC per project):
    python scripts/generate_missing_ttc_tiles.py --stats
"""

from __future__ import annotations

import argparse
import os
import sys

import duckdb

# Add src/ to path so we can import pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gri_tile_pipeline.tiles.csv_io import write_tiles_csv  # noqa: E402

HALF_TILE = 1.0 / 36  # half of 1/18 degree tile size


def print_stats(con: duckdb.DuckDBPyConnection, geoparquet: str) -> None:
    """Print full TTC coverage stats: existing + missing, per project and cohort.

    The ``ttc`` column is a MAP(year -> percent_cover).  A polygon "has ttc"
    when the map is non-NULL and non-empty; "correct year" means the map
    contains a key equal to ``YEAR(plantstart) - 1``.
    """
    # --- Per-project breakdown ---
    rows = con.execute(
        """
        SELECT
            short_name,
            framework_key,
            COUNT(*) AS total,
            SUM(CASE WHEN ttc IS NOT NULL AND cardinality(ttc) > 0
                     THEN 1 ELSE 0 END) AS with_ttc,
            SUM(CASE WHEN ttc IS NULL OR cardinality(ttc) = 0
                     THEN 1 ELSE 0 END) AS missing_ttc,
            SUM(CASE WHEN list_contains(map_keys(ttc), YEAR(plantstart) - 1)
                     THEN 1 ELSE 0 END) AS correct_yr,
            MIN(YEAR(plantstart) - 1) AS pred_yr_min,
            MAX(YEAR(plantstart) - 1) AS pred_yr_max
        FROM read_parquet($1)
        GROUP BY 1, 2
        ORDER BY missing_ttc DESC, short_name
        """,
        [geoparquet],
    ).fetchall()

    hdr = (
        f"  {'short_name':<30} {'framework_key':<25} "
        f"{'total':>7} {'w/ttc':>7} {'miss':>7} {'cor_yr':>7} "
        f"{'pred_yr':>11}"
    )
    print("TTC coverage by project:\n")
    print(hdr)
    print(
        f"  {'-'*30} {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7} "
        f"{'-'*11}"
    )
    for name, key, total, with_ttc, missing, correct_yr, yr_min, yr_max in rows:
        yr_range = f"{yr_min}-{yr_max}" if yr_min and yr_max else "n/a"
        print(
            f"  {(name or '(null)'):<30} {(key or ''):<25} "
            f"{total:>7,} {with_ttc:>7,} {missing:>7,} {correct_yr:>7,} "
            f"{yr_range:>11}"
        )

    # --- Cohort-level rollup ---
    cohort_rows = con.execute(
        """
        SELECT
            framework_key,
            COUNT(*) AS total,
            SUM(CASE WHEN ttc IS NOT NULL AND cardinality(ttc) > 0
                     THEN 1 ELSE 0 END) AS with_ttc,
            SUM(CASE WHEN ttc IS NULL OR cardinality(ttc) = 0
                     THEN 1 ELSE 0 END) AS missing_ttc,
            SUM(CASE WHEN list_contains(map_keys(ttc), YEAR(plantstart) - 1)
                     THEN 1 ELSE 0 END) AS correct_yr
        FROM read_parquet($1)
        GROUP BY 1
        ORDER BY missing_ttc DESC
        """,
        [geoparquet],
    ).fetchall()

    print("\nTTC coverage by cohort (framework_key):\n")
    print(
        f"  {'framework_key':<30} {'total':>7} {'w/ttc':>7} "
        f"{'miss':>7} {'cor_yr':>7}"
    )
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for key, total, with_ttc, missing, correct_yr in cohort_rows:
        print(
            f"  {(key or '(null)'):<30} "
            f"{total:>7,} {with_ttc:>7,} {missing:>7,} {correct_yr:>7,}"
        )

    # --- Grand totals ---
    grand_total = sum(r[1] for r in cohort_rows)
    grand_with = sum(r[2] for r in cohort_rows)
    grand_missing = sum(r[3] for r in cohort_rows)
    grand_correct = sum(r[4] for r in cohort_rows)
    grand_pct = (grand_with / grand_total * 100) if grand_total else 0
    print(
        f"\n  {'TOTAL':<30} "
        f"{grand_total:>7,} {grand_with:>7,} {grand_missing:>7,} {grand_correct:>7,}"
        f"  ({grand_pct:.1f}% have ttc)"
    )


def print_summary(con: duckdb.DuckDBPyConnection, geoparquet: str) -> None:
    """Print a summary of projects/cohorts with missing ttc."""
    print("Polygons with missing ttc by cohort (framework_key):")
    print(f"  {'framework_key':<30} {'polygons':>9}")
    print(f"  {'-'*30} {'-'*9}")
    rows = con.execute(
        """
        SELECT framework_key, COUNT(*) as cnt
        FROM read_parquet($1)
        WHERE ttc IS NULL OR cardinality(ttc) = 0
        GROUP BY 1
        ORDER BY cnt DESC
        """,
        [geoparquet],
    ).fetchall()
    for key, cnt in rows:
        print(f"  {key or '(null)':<30} {cnt:>9}")
    total = sum(r[1] for r in rows)

    print("\nPolygons with missing ttc by project (short_name):")
    print(f"  {'short_name':<30} {'framework_key':<25} {'polygons':>9}")
    print(f"  {'-'*30} {'-'*25} {'-'*9}")
    rows = con.execute(
        """
        SELECT short_name, framework_key, COUNT(*) as cnt
        FROM read_parquet($1)
        WHERE ttc IS NULL OR cardinality(ttc) = 0
        GROUP BY 1, 2
        ORDER BY cnt DESC
        """,
        [geoparquet],
    ).fetchall()
    for name, key, cnt in rows:
        print(f"  {name or '(null)':<30} {key or '':<25} {cnt:>9}")

    print(f"\nTotal polygons missing ttc: {total:,}")


def generate_tiles(
    con: duckdb.DuckDBPyConnection,
    geoparquet: str,
    tiledb: str,
    short_name: str | None = None,
    framework_key: str | None = None,
) -> list[dict]:
    """Spatial join null-ttc polygons against tile grid.

    Returns deduplicated tile dicts sorted by year.
    """
    # Build WHERE clause
    conditions = ["(ttc IS NULL OR cardinality(ttc) = 0)", "ST_IsValid(geom)"]
    params = [geoparquet]
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV of tiles needed for polygons missing TTC values."
    )
    parser.add_argument(
        "--geoparquet",
        default="temp/tm.geoparquet",
        help="Path to TerraMatch geoparquet (default: temp/tm.geoparquet)",
    )
    parser.add_argument(
        "--tiledb",
        default="data/tiledb.parquet",
        help="Path to tiledb.parquet (default: data/tiledb.parquet)",
    )
    parser.add_argument(
        "--short-name",
        default=None,
        help="Filter by project short_name",
    )
    parser.add_argument(
        "--framework-key",
        default=None,
        help="Filter by cohort framework_key",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show full TTC coverage stats (existing + missing) per project and cohort.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Output CSV path. If omitted, prints summary only.",
    )
    args = parser.parse_args()

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    # Full stats mode
    if args.stats:
        print_stats(con, args.geoparquet)
        con.close()
        return

    # If no output requested, print summary and exit
    if args.output is None and args.short_name is None and args.framework_key is None:
        print_summary(con, args.geoparquet)
        con.close()
        return

    # Describe what we're filtering
    filters = []
    if args.short_name:
        filters.append(f"short_name={args.short_name}")
    if args.framework_key:
        filters.append(f"framework_key={args.framework_key}")
    filter_desc = ", ".join(filters) if filters else "all projects"

    print(f"Finding tiles for missing-ttc polygons ({filter_desc}) ...")
    tiles = generate_tiles(
        con, args.geoparquet, args.tiledb,
        short_name=args.short_name,
        framework_key=args.framework_key,
    )
    con.close()

    if not tiles:
        print("No tiles found matching the criteria.")
        return

    # Print summary by year
    year_counts: dict[int, int] = {}
    for t in tiles:
        year_counts[t["year"]] = year_counts.get(t["year"], 0) + 1
    print(f"\nUnique tiles: {len(tiles)}")
    for year in sorted(year_counts):
        print(f"  {year}: {year_counts[year]} tiles")

    if args.output:
        write_tiles_csv(args.output, tiles)
        print(f"\nWritten to {args.output}")
    else:
        print("\nUse --output PATH to write CSV.")


if __name__ == "__main__":
    main()
