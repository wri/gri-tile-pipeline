"""Generate a TTC status report across four phases.

Phases:
  1. Request Scope — what projects/polygons match the input
  2. TTC Coverage — how many already have TTC indicator values
  3. Tile Availability — do the needed prediction tiles exist on S3
  4. Tiles to Generate — final missing tile list

Returns structured data + a Markdown rendering. The CLI subcommand wraps this.
"""

from __future__ import annotations

import csv as csv_mod
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


_FORBIDDEN_SQL = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|ATTACH|COPY|PRAGMA)\b",
    re.IGNORECASE,
)


def _validate_where_sql(where_sql: str) -> None:
    """Reject statement-terminators and DDL/DML keywords. Read path is SELECT-only."""
    if ";" in where_sql:
        raise ValueError("--where clause must not contain ';'")
    if _FORBIDDEN_SQL.search(where_sql):
        raise ValueError(
            "--where clause rejected: contains DDL/DML keyword "
            "(DROP, DELETE, INSERT, UPDATE, ALTER, CREATE, TRUNCATE, GRANT, REVOKE, ATTACH, COPY, PRAGMA)"
        )

from gri_tile_pipeline.duckdb_utils import connect_with_spatial
from gri_tile_pipeline.tiles.csv_io import write_tiles_csv


HALF_TILE = 1.0 / 36


@dataclass
class ReportResult:
    scope: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    tile_avail: dict | None = None
    tiles_csv_path: str | None = None
    report_path: str | None = None
    report_markdown: str = ""
    filter_desc: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Phase 1: Request Scope
# ---------------------------------------------------------------------------

def _build_filter(
    con,
    geoparquet: str,
    input_csv: str | None,
    project_ids: list[str],
    short_names: list[str],
    framework_keys: list[str],
    poly_uuids: list[str] | None = None,
    cohorts: list[str] | None = None,
    where_sql: str | None = None,
) -> tuple[str, list]:
    conditions: list[str] = []
    params: list = [geoparquet]
    idx = 2
    poly_uuids = poly_uuids or []
    cohorts = cohorts or []

    if input_csv:
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

    if poly_uuids:
        placeholders = ", ".join(f"${idx + i}" for i in range(len(poly_uuids)))
        conditions.append(f"p.poly_uuid IN ({placeholders})")
        params.extend(poly_uuids)
        idx += len(poly_uuids)

    if cohorts:
        # cohort is a BLOB-encoded list; match if any requested value is contained.
        clauses = []
        for c in cohorts:
            clauses.append(f"list_contains(p.cohort, ${idx})")
            params.append(c)
            idx += 1
        conditions.append("(" + " OR ".join(clauses) + ")")

    if where_sql:
        _validate_where_sql(where_sql)
        conditions.append(f"({where_sql})")

    if not conditions:
        raise ValueError(
            "Provide at least one of: input_csv, project_ids, short_names, "
            "framework_keys, poly_uuids, cohorts, where_sql."
        )

    return " AND ".join(conditions), params


def resolve_scope(
    con, geoparquet, input_csv, project_ids, short_names, framework_keys,
    poly_uuids=None, cohorts=None, where_sql=None,
) -> tuple[dict, str, list]:
    where, params = _build_filter(
        con, geoparquet, input_csv, project_ids, short_names, framework_keys,
        poly_uuids=poly_uuids, cohorts=cohorts, where_sql=where_sql,
    )
    projects = con.execute(
        f"""
        SELECT
            p.project_id, p.short_name, p.framework_key,
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
    return {
        "total_projects": len(projects),
        "total_polygons": total_polys,
        "projects": [
            {
                "project_id": r[0], "short_name": r[1] or "(none)",
                "framework_key": r[2] or "(none)", "n_polys": r[3],
                "yr_min": r[4], "yr_max": r[5], "area_ha": r[6],
            }
            for r in projects
        ],
    }, where, params


# ---------------------------------------------------------------------------
# Phase 2: TTC Coverage
# ---------------------------------------------------------------------------

def check_ttc_coverage(con, geoparquet: str, where: str, params: list) -> dict:
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
    return {
        "total": sum(r[1] for r in rows),
        "correct_yr": sum(r[2] for r in rows),
        "wrong_yr": sum(r[3] for r in rows),
        "missing": sum(r[4] for r in rows),
        "per_project": [
            {"label": r[0], "total": r[1], "correct_yr": r[2],
             "wrong_yr": r[3], "missing": r[4]}
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# Phase 3: Tile Availability
# ---------------------------------------------------------------------------

def find_needed_tiles(con, geoparquet: str, tiledb: str, where: str, params: list) -> list[dict]:
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
            q.yr AS year, t.X AS lon, t.Y AS lat,
            t.X_tile, t.Y_tile
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
        {"year": int(y), "lon": round(float(lon), 4), "lat": round(float(lat), 4),
         "X_tile": int(xt), "Y_tile": int(yt)}
        for y, lon, lat, xt, yt in rows
    ]


def check_s3(
    tiles: list[dict], bucket: str, region: str,
    aws_profile: str | None, check_type: str,
) -> dict:
    """Check which tiles exist on S3. Uses obstore_utils.make_s3_store + validate_aws."""
    from gri_tile_pipeline.storage.obstore_utils import make_s3_store, validate_aws
    from gri_tile_pipeline.tiles.availability import check_availability

    store = make_s3_store(bucket, region=region, profile=aws_profile)
    validate_aws(store)

    result = check_availability(
        tiles, f"s3://{bucket}", check_type=check_type, region=region, store=store,
    )
    existing = result["existing"]
    missing = result["missing"]

    per_year: dict[int, dict] = {}
    for t in tiles:
        yr = t["year"]
        per_year.setdefault(yr, {"needed": 0, "existing": 0, "missing": 0})
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
# Rendering
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list], alignments: list[str] | None = None) -> str:
    if not alignments:
        alignments = ["l"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |"]
    sep = []
    for a in alignments:
        sep.append("---:" if a == "r" else (":---:" if a == "c" else "---"))
    lines.append("| " + " | ".join(sep) + " |")
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
    lines = ["# TTC Status Report", f"Generated: {timestamp}", ""]

    # Phase 1
    lines.append("## 1. Request Scope")
    lines.append(f"**Filter**: {filter_desc}")
    lines.append(
        f"**Projects**: {scope['total_projects']} | "
        f"**Polygons**: {scope['total_polygons']:,}"
    )
    lines.append("")
    rows = []
    for p in scope["projects"]:
        yr_range = f"{p['yr_min']}-{p['yr_max']}" if p["yr_min"] != p["yr_max"] else str(p["yr_min"])
        rows.append([
            (p["project_id"] or "")[:12] + "...",
            p["short_name"], p["framework_key"],
            f"{p['n_polys']:,}", yr_range,
            f"{p['area_ha']:,.1f}" if p["area_ha"] else "-",
        ])
    lines.append(_md_table(
        ["project_id", "short_name", "framework_key", "polygons", "plantstart", "area_ha"],
        rows, ["l", "l", "l", "r", "c", "r"],
    ))
    lines.append("")

    # Phase 2
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
            p["label"], f"{p['total']:,}", f"{p['correct_yr']:,}",
            f"{p['wrong_yr']:,}", f"{p['missing']:,}", f"{p_pct:.0f}%",
        ])
    lines.append(_md_table(
        ["project", "total", "correct_yr", "wrong_yr", "missing", "coverage"],
        rows, ["l", "r", "r", "r", "r", "r"],
    ))
    lines.append("")

    # Phase 3
    lines.append("## 3. Tile Availability")
    s3_checked = tile_avail is not None and tile_avail.get("existing") is not None

    if tile_avail is None:
        lines.append("*(skipped — run without --skip-s3 to check S3)*")
    elif tile_avail["total_needed"] == 0:
        lines.append("No tiles needed — all polygons have TTC data.")
    elif not s3_checked:
        lines.append(f"**Tiles needed**: {tile_avail['total_needed']} *(S3 check skipped)*")
        lines.append("")
        rows = [[str(yr), str(d["needed"])]
                for yr, d in sorted(tile_avail["per_year"].items())]
        lines.append(_md_table(["year", "needed"], rows, ["l", "r"]))
    else:
        total_n = tile_avail["total_needed"]
        ex = tile_avail["existing"]
        mi = tile_avail["missing"]
        pct_s3 = (ex / total_n * 100) if total_n else 0
        lines.append(
            f"**Tiles needed**: {total_n} | "
            f"**On S3**: {ex} ({pct_s3:.1f}%) | "
            f"**Missing**: {mi}"
        )
        lines.append("")
        rows = [[str(yr), str(d["needed"]), str(d["existing"]), str(d["missing"])]
                for yr, d in sorted(tile_avail["per_year"].items())]
        lines.append(_md_table(
            ["year", "needed", "existing", "missing"],
            rows, ["l", "r", "r", "r"],
        ))
    lines.append("")

    # Phase 4
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
        rows = [[str(yr), str(d["missing"])]
                for yr, d in sorted(tile_avail["per_year"].items())
                if d["missing"] > 0]
        if rows:
            lines.append(_md_table(["year", "tiles_to_generate"], rows, ["l", "r"]))

    lines.append("")
    return "\n".join(lines)


def _filter_description(
    input_csv, project_ids, short_names, framework_keys,
    poly_uuids=None, cohorts=None, where_sql=None,
) -> str:
    parts = []
    if input_csv:
        with open(input_csv) as f:
            n_rows = sum(1 for _ in csv_mod.DictReader(f))
        parts.append(f"CSV `{os.path.basename(input_csv)}` ({n_rows} rows)")
    if project_ids:
        parts.append(
            f"project_id in [{', '.join(p[:12] + '...' for p in project_ids)}]"
        )
    if short_names:
        parts.append(f"short_name in [{', '.join(short_names)}]")
    if framework_keys:
        parts.append(f"framework_key in [{', '.join(framework_keys)}]")
    if poly_uuids:
        if len(poly_uuids) <= 3:
            parts.append(f"poly_uuid in [{', '.join(poly_uuids)}]")
        else:
            parts.append(f"poly_uuid in [{len(poly_uuids)} uuids]")
    if cohorts:
        parts.append(f"cohort contains any of [{', '.join(cohorts)}]")
    if where_sql:
        parts.append(f"where `{where_sql}`")
    return " + ".join(parts)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def generate_report(
    *,
    input_csv: str | None = None,
    project_ids: list[str] | None = None,
    short_names: list[str] | None = None,
    framework_keys: list[str] | None = None,
    poly_uuids: list[str] | None = None,
    cohorts: list[str] | None = None,
    where_sql: str | None = None,
    geoparquet: str = "temp/tm.geoparquet",
    tiledb: str = "data/tiledb.parquet",
    bucket: str = "tof-output",
    region: str = "us-east-1",
    check_type: str = "predictions",
    skip_s3: bool = False,
    aws_profile: str | None = None,
    output_dir: str = ".",
    progress=None,  # optional callable(str) for phase messages
) -> ReportResult:
    """Generate the 4-phase TTC status report and write a Markdown file to *output_dir*."""
    project_ids = project_ids or []
    short_names = short_names or []
    framework_keys = framework_keys or []
    poly_uuids = poly_uuids or []
    cohorts = cohorts or []

    def _log(msg: str) -> None:
        if progress:
            progress(msg)

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d %H:%M UTC")
    file_ts = now.strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    filter_desc = _filter_description(
        input_csv, project_ids, short_names, framework_keys,
        poly_uuids=poly_uuids, cohorts=cohorts, where_sql=where_sql,
    )
    result = ReportResult(filter_desc=filter_desc, timestamp=timestamp)

    con = connect_with_spatial()
    try:
        _log("Phase 1/4: Resolving request scope...")
        scope, where, params = resolve_scope(
            con, geoparquet, input_csv, project_ids, short_names, framework_keys,
            poly_uuids=poly_uuids, cohorts=cohorts, where_sql=where_sql,
        )
        result.scope = scope
        if scope["total_polygons"] == 0:
            raise ValueError("No matching polygons found. Check your filters.")

        _log("Phase 2/4: Checking TTC coverage...")
        coverage = check_ttc_coverage(con, geoparquet, where, params)
        result.coverage = coverage

        tile_avail = None
        tiles_csv_path = None

        if skip_s3:
            _log("Phase 3/4: Skipped (--skip-s3)")
            if coverage["missing"] > 0:
                needed_tiles = find_needed_tiles(con, geoparquet, tiledb, where, params)
                if needed_tiles:
                    year_counts: dict[int, int] = {}
                    for t in needed_tiles:
                        year_counts[t["year"]] = year_counts.get(t["year"], 0) + 1
                    tiles_csv_path = os.path.join(
                        output_dir, f"ttc_needed_tiles_{file_ts}.csv",
                    )
                    write_tiles_csv(tiles_csv_path, needed_tiles)
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
            _log("Phase 3/4: Checking tile availability on S3...")
            if coverage["missing"] == 0:
                tile_avail = {"total_needed": 0, "existing": 0, "missing": 0, "per_year": {}}
            else:
                needed_tiles = find_needed_tiles(con, geoparquet, tiledb, where, params)
                if not needed_tiles:
                    tile_avail = {"total_needed": 0, "existing": 0, "missing": 0, "per_year": {}}
                else:
                    tile_avail = check_s3(
                        needed_tiles, bucket, region, aws_profile, check_type,
                    )
                    if tile_avail["missing"] > 0:
                        tiles_csv_path = os.path.join(
                            output_dir, f"ttc_missing_tiles_{file_ts}.csv",
                        )
                        write_tiles_csv(tiles_csv_path, tile_avail["missing_tiles"])

        result.tile_avail = tile_avail
        result.tiles_csv_path = tiles_csv_path
    finally:
        con.close()

    result.report_markdown = render_report(
        scope, coverage, tile_avail, tiles_csv_path, filter_desc, timestamp,
    )
    report_path = os.path.join(output_dir, f"ttc_status_{file_ts}.md")
    with open(report_path, "w") as f:
        f.write(result.report_markdown)
    result.report_path = report_path

    return result
