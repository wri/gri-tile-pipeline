#!/usr/bin/env python
"""Generate a TTC status report for a set of projects/polygons.

Runs four phases and produces a shareable Markdown report:
  1. Request Scope — what projects/polygons match the input
  2. TTC Coverage — how many already have TTC indicator values
  3. Tile Availability — do the needed prediction tiles exist on S3
  4. Tiles to Generate — final missing tile list (CSV)

Usage:
    # From a request CSV (phases 1-2 only, no AWS needed):
    python scripts/ttc_status_report.py \
        --input example/status_request.csv --skip-s3

    # Full report with S3 check:
    python scripts/ttc_status_report.py \
        --input example/status_request.csv --aws-profile <profile>

    # Filter by framework_key instead of CSV:
    python scripts/ttc_status_report.py \
        --framework-key terrafund-landscapes --skip-s3

    # Filter by short_name:
    python scripts/ttc_status_report.py \
        --short-name RWA_23_AEE --skip-s3
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone

import duckdb

# Add src/ to path so we can import pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gri_tile_pipeline.tiles.csv_io import write_tiles_csv  # noqa: E402

HALF_TILE = 1.0 / 36  # half of 1/18 degree tile size


# ---------------------------------------------------------------------------
# Phase 1: Request Scope
# ---------------------------------------------------------------------------

def _build_filter(
    con: duckdb.DuckDBPyConnection,
    geoparquet: str,
    input_csv: str | None,
    project_ids: list[str],
    short_names: list[str],
    framework_keys: list[str],
) -> tuple[str, list]:
    """Build a DuckDB WHERE clause from the input options.

    Returns (where_clause, params) where $1 is always the geoparquet path.
    """
    conditions: list[str] = []
    params: list = [geoparquet]
    idx = 2  # $1 is geoparquet

    if input_csv:
        # Register CSV as a DuckDB table and join on project_id + optional year
        con.execute(
            "CREATE OR REPLACE TEMP TABLE request_input AS "
            "SELECT * FROM read_csv_auto($1, header=true)",
            [input_csv],
        )
        cols = [r[0] for r in con.execute("DESCRIBE request_input").fetchall()]
        has_year = "plantstart_year" in cols

        if has_year:
            conditions.append(
                "EXISTS ("
                "  SELECT 1 FROM request_input r"
                "  WHERE r.project_id = p.project_id"
                "    AND (r.plantstart_year IS NULL OR r.plantstart_year = YEAR(p.plantstart))"
                ")"
            )
        else:
            conditions.append(
                "EXISTS ("
                "  SELECT 1 FROM request_input r"
                "  WHERE r.project_id = p.project_id"
                ")"
            )

    if project_ids:
        placeholders = ", ".join(f"${idx + i}" for i in range(len(project_ids)))
        conditions.append(f"p.project_id IN ({placeholders})")
        params.extend(project_ids)
        idx += len(project_ids)

    if short_names:
        placeholders = ", ".join(f"${idx + i}" for i in range(len(short_names)))
        conditions.append(f"p.short_name IN ({placeholders})")
        params.extend(short_names)
        idx += len(short_names)

    if framework_keys:
        placeholders = ", ".join(f"${idx + i}" for i in range(len(framework_keys)))
        conditions.append(f"p.framework_key IN ({placeholders})")
        params.extend(framework_keys)
        idx += len(framework_keys)

    if not conditions:
        print("Error: provide --input, --project-id, --short-name, or --framework-key")
        sys.exit(1)

    where = " AND ".join(conditions)
    return where, params


def resolve_scope(
    con: duckdb.DuckDBPyConnection,
    geoparquet: str,
    input_csv: str | None,
    project_ids: list[str],
    short_names: list[str],
    framework_keys: list[str],
) -> tuple[dict, str, list]:
    """Phase 1: resolve the request into matching polygons.

    Returns (scope_dict, where_clause, params).
    """
    where, params = _build_filter(
        con, geoparquet, input_csv, project_ids, short_names, framework_keys,
    )

    # Project-level summary
    projects = con.execute(
        f"""
        SELECT
            p.project_id,
            p.short_name,
            p.framework_key,
            COUNT(*) AS n_polys,
            MIN(YEAR(p.plantstart)) AS yr_min,
            MAX(YEAR(p.plantstart)) AS yr_max,
            ROUND(SUM(p.calc_area), 1) AS total_area
        FROM read_parquet($1) p
        WHERE {where}
        GROUP BY 1, 2, 3
        ORDER BY n_polys DESC
        """,
        params,
    ).fetchall()

    total_polys = sum(r[3] for r in projects)

    scope = {
        "total_projects": len(projects),
        "total_polygons": total_polys,
        "projects": [
            {
                "project_id": r[0],
                "short_name": r[1] or "(none)",
                "framework_key": r[2] or "(none)",
                "n_polys": r[3],
                "yr_min": r[4],
                "yr_max": r[5],
                "area_ha": r[6],
            }
            for r in projects
        ],
    }

    return scope, where, params


# ---------------------------------------------------------------------------
# Phase 2: TTC Coverage
# ---------------------------------------------------------------------------

def check_ttc_coverage(
    con: duckdb.DuckDBPyConnection,
    geoparquet: str,
    where: str,
    params: list,
) -> dict:
    """Phase 2: check TTC coverage for the scoped polygons."""
    rows = con.execute(
        f"""
        SELECT
            COALESCE(p.short_name, p.project_id) AS label,
            COUNT(*) AS total,
            SUM(CASE WHEN p.ttc IS NOT NULL AND cardinality(p.ttc) > 0
                      AND list_contains(map_keys(p.ttc), YEAR(p.plantstart) - 1)
                     THEN 1 ELSE 0 END) AS correct_yr,
            SUM(CASE WHEN p.ttc IS NOT NULL AND cardinality(p.ttc) > 0
                      AND NOT list_contains(map_keys(p.ttc), YEAR(p.plantstart) - 1)
                     THEN 1 ELSE 0 END) AS wrong_yr,
            SUM(CASE WHEN p.ttc IS NULL OR cardinality(p.ttc) = 0
                     THEN 1 ELSE 0 END) AS missing
        FROM read_parquet($1) p
        WHERE {where}
        GROUP BY 1
        ORDER BY missing DESC, 1
        """,
        params,
    ).fetchall()

    total = sum(r[1] for r in rows)
    correct = sum(r[2] for r in rows)
    wrong = sum(r[3] for r in rows)
    missing = sum(r[4] for r in rows)

    return {
        "total": total,
        "correct_yr": correct,
        "wrong_yr": wrong,
        "missing": missing,
        "per_project": [
            {
                "label": r[0],
                "total": r[1],
                "correct_yr": r[2],
                "wrong_yr": r[3],
                "missing": r[4],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# Phase 3: Tile Availability
# ---------------------------------------------------------------------------

def find_needed_tiles(
    con: duckdb.DuckDBPyConnection,
    geoparquet: str,
    tiledb: str,
    where: str,
    params: list,
) -> list[dict]:
    """Spatial join missing-TTC polygons against the tile grid.

    Returns deduplicated tile dicts sorted by year.
    """
    # Add tiledb as the next parameter
    tiledb_idx = len(params) + 1
    tile_params = params + [tiledb]

    rows = con.execute(
        f"""
        WITH polys AS (
            SELECT p.geom, YEAR(p.plantstart) - 1 AS yr
            FROM read_parquet($1) p
            WHERE {where}
              AND (p.ttc IS NULL OR cardinality(p.ttc) = 0)
              AND ST_IsValid(p.geom)
        )
        SELECT DISTINCT
            q.yr AS year,
            t.X AS lon,
            t.Y AS lat,
            t.X_tile,
            t.Y_tile
        FROM polys q, read_parquet(${tiledb_idx}) t
        WHERE ST_Intersects(
            q.geom,
            ST_MakeEnvelope(
                t.X - {HALF_TILE}, t.Y - {HALF_TILE},
                t.X + {HALF_TILE}, t.Y + {HALF_TILE}
            )
        )
        ORDER BY q.yr, t.X_tile, t.Y_tile
        """,
        tile_params,
    ).fetchall()

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


def check_s3(
    tiles: list[dict],
    bucket: str,
    region: str,
    aws_profile: str | None,
    check_type: str,
) -> dict:
    """Check which tiles exist on S3.

    Returns {"existing": [...], "missing": [...], "per_year": {...}}.
    """
    import boto3
    from obstore.auth.boto3 import Boto3CredentialProvider
    from obstore.store import S3Store

    from gri_tile_pipeline.tiles.availability import check_availability

    session = boto3.Session(profile_name=aws_profile)
    credential_provider = Boto3CredentialProvider(session)
    store = S3Store(bucket, region=region, credential_provider=credential_provider)

    # Validate credentials
    import obstore as obs
    try:
        obs.head(store, "__credential_check__")
    except FileNotFoundError:
        pass
    except Exception as e:
        first_line = str(e).split("\n")[0]
        print(f"AWS credential check failed: {first_line}")
        print("Use --aws-profile <profile> or aws sso login")
        sys.exit(1)

    result = check_availability(
        tiles, f"s3://{bucket}",
        check_type=check_type, region=region,
        store=store,
    )
    existing = result["existing"]
    missing = result["missing"]

    # Per-year breakdown
    per_year: dict[int, dict] = {}
    for t in tiles:
        yr = t["year"]
        if yr not in per_year:
            per_year[yr] = {"needed": 0, "existing": 0, "missing": 0}
        per_year[yr]["needed"] += 1

    for t in existing:
        per_year[t["year"]]["existing"] += 1
    for t in missing:
        per_year[t["year"]]["missing"] += 1

    return {
        "total_needed": len(tiles),
        "existing": len(existing),
        "missing": len(missing),
        "existing_tiles": existing,
        "missing_tiles": missing,
        "per_year": per_year,
    }


# ---------------------------------------------------------------------------
# Report Rendering
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list], alignments: list[str] | None = None) -> str:
    """Render a Markdown table."""
    if not alignments:
        alignments = ["l"] * len(headers)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    sep_parts = []
    for a in alignments:
        if a == "r":
            sep_parts.append("---:")
        elif a == "c":
            sep_parts.append(":---:")
        else:
            sep_parts.append("---")
    lines.append("| " + " | ".join(sep_parts) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def render_report(
    scope: dict,
    coverage: dict,
    tile_avail: dict | None,
    tiles_csv_path: str | None,
    filter_desc: str,
    timestamp: str,
) -> str:
    """Render the full Markdown report."""
    lines = [
        "# TTC Status Report",
        f"Generated: {timestamp}",
        "",
    ]

    # --- Phase 1 ---
    lines.append("## 1. Request Scope")
    lines.append(f"**Filter**: {filter_desc}")
    lines.append(
        f"**Projects**: {scope['total_projects']} | "
        f"**Polygons**: {scope['total_polygons']:,}"
    )
    lines.append("")

    rows = []
    for p in scope["projects"]:
        yr_range = (
            f"{p['yr_min']}-{p['yr_max']}" if p["yr_min"] != p["yr_max"]
            else str(p["yr_min"])
        )
        rows.append([
            p["project_id"][:12] + "...",
            p["short_name"],
            p["framework_key"],
            f"{p['n_polys']:,}",
            yr_range,
            f"{p['area_ha']:,.1f}" if p["area_ha"] else "-",
        ])

    lines.append(_md_table(
        ["project_id", "short_name", "framework_key", "polygons", "plantstart", "area_ha"],
        rows,
        ["l", "l", "l", "r", "c", "r"],
    ))
    lines.append("")

    # --- Phase 2 ---
    lines.append("## 2. TTC Coverage")

    total = coverage["total"]
    correct = coverage["correct_yr"]
    pct = (correct / total * 100) if total else 0
    lines.append(
        f"**With correct year**: {correct:,} / {total:,} ({pct:.1f}%) | "
        f"**Wrong year**: {coverage['wrong_yr']:,} | "
        f"**Missing**: {coverage['missing']:,}"
    )
    lines.append("")

    rows = []
    for p in coverage["per_project"]:
        p_pct = (p["correct_yr"] / p["total"] * 100) if p["total"] else 0
        rows.append([
            p["label"],
            f"{p['total']:,}",
            f"{p['correct_yr']:,}",
            f"{p['wrong_yr']:,}",
            f"{p['missing']:,}",
            f"{p_pct:.0f}%",
        ])

    lines.append(_md_table(
        ["project", "total", "correct_yr", "wrong_yr", "missing", "coverage"],
        rows,
        ["l", "r", "r", "r", "r", "r"],
    ))
    lines.append("")

    # --- Phase 3 ---
    lines.append("## 3. Tile Availability")

    s3_checked = tile_avail is not None and tile_avail["existing"] is not None

    if tile_avail is None:
        lines.append("*(skipped — run without --skip-s3 to check S3)*")
    elif tile_avail["total_needed"] == 0:
        lines.append("No tiles needed — all polygons have TTC data.")
    elif not s3_checked:
        # skip-s3 mode: we know tiles needed but not S3 status
        total_n = tile_avail["total_needed"]
        lines.append(f"**Tiles needed**: {total_n} *(S3 check skipped)*")
        lines.append("")

        rows = []
        for yr in sorted(tile_avail["per_year"]):
            d = tile_avail["per_year"][yr]
            rows.append([str(yr), str(d["needed"])])

        lines.append(_md_table(
            ["year", "needed"],
            rows,
            ["l", "r"],
        ))
    else:
        total_n = tile_avail["total_needed"]
        ex = tile_avail["existing"]
        mi = tile_avail["missing"]
        pct = (ex / total_n * 100) if total_n else 0
        lines.append(
            f"**Tiles needed**: {total_n} | "
            f"**On S3**: {ex} ({pct:.1f}%) | "
            f"**Missing**: {mi}"
        )
        lines.append("")

        rows = []
        for yr in sorted(tile_avail["per_year"]):
            d = tile_avail["per_year"][yr]
            rows.append([str(yr), str(d["needed"]), str(d["existing"]), str(d["missing"])])

        lines.append(_md_table(
            ["year", "needed", "existing", "missing"],
            rows,
            ["l", "r", "r", "r"],
        ))

    lines.append("")

    # --- Phase 4 ---
    lines.append("## 4. Tiles to Generate")

    if tile_avail is None:
        lines.append("*(skipped — S3 check not run)*")
    elif not s3_checked:
        lines.append(f"**{tile_avail['total_needed']} tiles** need S3 availability check.")
        if tiles_csv_path:
            lines.append(f"**Needed tiles CSV**: `{tiles_csv_path}`")
    elif tile_avail["missing"] == 0:
        lines.append("All needed tiles already exist on S3.")
    else:
        lines.append(f"**Total**: {tile_avail['missing']} tiles")
        if tiles_csv_path:
            lines.append(f"**CSV**: `{tiles_csv_path}`")
        lines.append("")

        # Summary by year
        rows = []
        for yr in sorted(tile_avail["per_year"]):
            d = tile_avail["per_year"][yr]
            if d["missing"] > 0:
                rows.append([str(yr), str(d["missing"])])
        if rows:
            lines.append(_md_table(["year", "tiles_to_generate"], rows, ["l", "r"]))

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _filter_description(input_csv, project_ids, short_names, framework_keys) -> str:
    parts = []
    if input_csv:
        # Count rows in CSV
        with open(input_csv) as f:
            reader = csv.DictReader(f)
            n_rows = sum(1 for _ in reader)
        parts.append(f"CSV `{os.path.basename(input_csv)}` ({n_rows} rows)")
    if project_ids:
        parts.append(f"project_id in [{', '.join(p[:12] + '...' for p in project_ids)}]")
    if short_names:
        parts.append(f"short_name in [{', '.join(short_names)}]")
    if framework_keys:
        parts.append(f"framework_key in [{', '.join(framework_keys)}]")
    return " + ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a TTC status report for a set of projects/polygons."
    )
    parser.add_argument(
        "--input", dest="input_csv", default=None,
        help="CSV with project_id[,plantstart_year] pairs.",
    )
    parser.add_argument(
        "--project-id", dest="project_ids", action="append", default=[],
        help="Filter by project_id (repeatable).",
    )
    parser.add_argument(
        "--short-name", dest="short_names", action="append", default=[],
        help="Filter by short_name (repeatable).",
    )
    parser.add_argument(
        "--framework-key", dest="framework_keys", action="append", default=[],
        help="Filter by framework_key (repeatable).",
    )
    parser.add_argument(
        "--geoparquet", default="temp/tm.geoparquet",
        help="Path to TerraMatch geoparquet (default: temp/tm.geoparquet).",
    )
    parser.add_argument(
        "--tiledb", default="data/tiledb.parquet",
        help="Path to tiledb.parquet (default: data/tiledb.parquet).",
    )
    parser.add_argument(
        "--bucket", default="tof-output",
        help="S3 bucket (default: tof-output).",
    )
    parser.add_argument(
        "--region", default="us-east-1",
        help="S3 bucket region (default: us-east-1).",
    )
    parser.add_argument(
        "--check-type", choices=["predictions", "raw_ard"], default="predictions",
        help="What to check on S3 (default: predictions).",
    )
    parser.add_argument(
        "--skip-s3", action="store_true",
        help="Skip S3 availability check (phases 3-4).",
    )
    parser.add_argument(
        "--aws-profile", default=None,
        help="AWS profile name for S3 access.",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory for output files (default: current directory).",
    )
    args = parser.parse_args()

    if not args.input_csv and not args.project_ids and not args.short_names and not args.framework_keys:
        parser.error("Provide at least one of: --input, --project-id, --short-name, --framework-key")

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d %H:%M UTC")
    file_ts = now.strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)

    filter_desc = _filter_description(
        args.input_csv, args.project_ids, args.short_names, args.framework_keys,
    )

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    # --- Phase 1 ---
    print(f"Phase 1/4: Resolving request scope...")
    scope, where, params = resolve_scope(
        con, args.geoparquet, args.input_csv,
        args.project_ids, args.short_names, args.framework_keys,
    )
    print(f"  {scope['total_projects']} projects, {scope['total_polygons']:,} polygons")

    if scope["total_polygons"] == 0:
        print("No matching polygons found. Check your filters.")
        con.close()
        sys.exit(1)

    # --- Phase 2 ---
    print(f"Phase 2/4: Checking TTC coverage...")
    coverage = check_ttc_coverage(con, args.geoparquet, where, params)
    pct = (coverage["correct_yr"] / coverage["total"] * 100) if coverage["total"] else 0
    print(f"  {coverage['correct_yr']:,}/{coverage['total']:,} ({pct:.1f}%) have correct year")
    print(f"  {coverage['missing']:,} missing TTC entirely")

    # --- Phases 3 & 4 ---
    tile_avail = None
    tiles_csv_path = None

    if args.skip_s3:
        print("Phase 3/4: Skipped (--skip-s3)")
        print("Phase 4/4: Skipped")

        # Still do the spatial join to report how many tiles are needed
        if coverage["missing"] > 0:
            print(f"\n  Finding tiles for {coverage['missing']:,} missing-TTC polygons...")
            needed_tiles = find_needed_tiles(
                con, args.geoparquet, args.tiledb, where, params,
            )
            if needed_tiles:
                year_counts: dict[int, int] = {}
                for t in needed_tiles:
                    year_counts[t["year"]] = year_counts.get(t["year"], 0) + 1
                print(f"  {len(needed_tiles)} unique tiles needed")
                for yr in sorted(year_counts):
                    print(f"    {yr}: {year_counts[yr]} tiles")

                # Write tiles CSV even in skip-s3 mode (all needed, not yet filtered)
                tiles_csv_path = os.path.join(
                    args.output_dir, f"ttc_needed_tiles_{file_ts}.csv",
                )
                write_tiles_csv(tiles_csv_path, needed_tiles)
                print(f"  Needed tiles CSV: {tiles_csv_path}")

                tile_avail = {
                    "total_needed": len(needed_tiles),
                    "existing": None,
                    "missing": None,
                    "per_year": {
                        yr: {"needed": c, "existing": "?", "missing": "?"}
                        for yr, c in year_counts.items()
                    },
                }
    else:
        print(f"Phase 3/4: Checking tile availability on S3...")
        if coverage["missing"] == 0:
            print("  No polygons missing TTC — skipping tile check.")
            tile_avail = {"total_needed": 0, "existing": 0, "missing": 0, "per_year": {}}
        else:
            needed_tiles = find_needed_tiles(
                con, args.geoparquet, args.tiledb, where, params,
            )
            if not needed_tiles:
                print("  No valid tiles found (invalid geometries?).")
                tile_avail = {"total_needed": 0, "existing": 0, "missing": 0, "per_year": {}}
            else:
                print(f"  {len(needed_tiles)} unique tiles needed, checking S3...")
                tile_avail = check_s3(
                    needed_tiles, args.bucket, args.region,
                    args.aws_profile, args.check_type,
                )
                pct_s3 = (tile_avail["existing"] / tile_avail["total_needed"] * 100) if tile_avail["total_needed"] else 0
                print(f"  {tile_avail['existing']}/{tile_avail['total_needed']} exist ({pct_s3:.1f}%)")
                print(f"  {tile_avail['missing']} tiles to generate")

                # Phase 4: write missing tiles CSV
                if tile_avail["missing"] > 0:
                    tiles_csv_path = os.path.join(
                        args.output_dir, f"ttc_missing_tiles_{file_ts}.csv",
                    )
                    write_tiles_csv(tiles_csv_path, tile_avail["missing_tiles"])
                    print(f"Phase 4/4: Missing tiles CSV: {tiles_csv_path}")
                else:
                    print("Phase 4/4: All tiles exist — nothing to generate.")

    con.close()

    # --- Render report ---
    report_md = render_report(
        scope, coverage, tile_avail, tiles_csv_path, filter_desc, timestamp,
    )
    report_path = os.path.join(args.output_dir, f"ttc_status_{file_ts}.md")
    with open(report_path, "w") as f:
        f.write(report_md)

    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
