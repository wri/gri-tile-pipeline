#!/usr/bin/env python
"""Identify polygons that were dropped during a pipeline run and diagnose why.

Compares the set of polygons that *should* have been processed (from the
request CSV + geoparquet) against the output stats CSV to find which
poly_uuids are missing, then classifies the geometry issue for each.

Usage:
    uv run python scripts/audit_dropped_polygons.py \
        --request temp/missing_ttc_request_cohort_2.csv \
        --stats temp/62_projects_request_stats.csv

    uv run python scripts/audit_dropped_polygons.py \
        --request temp/missing_ttc_request_cohort_2.csv \
        --stats temp/62_projects_request_stats.csv \
        -o temp/dropped_polygons_report.csv
"""
from __future__ import annotations

import argparse
import csv
import sys

import duckdb
import pandas as pd
from loguru import logger

GEOPARQUET = "temp/tm.geoparquet"


def load_expected_polygons(request_csv: str, geoparquet: str) -> pd.DataFrame:
    """Load all poly_uuids that match the request CSV from the geoparquet."""
    req = pd.read_csv(request_csv)
    if not {"project_id", "plantstart_year"}.issubset(req.columns):
        logger.error("Request CSV must have 'project_id' and 'plantstart_year' columns")
        sys.exit(1)

    conditions = " OR ".join(
        f"(project_id = '{row.project_id}' AND YEAR(plantstart) = {int(row.plantstart_year)})"
        for _, row in req.iterrows()
    )

    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")

    df = con.sql(f"""
        SELECT
            poly_uuid,
            project_id,
            short_name,
            YEAR(plantstart) AS plantstart_year,
            ST_AsWKB(geom) AS __wkb,
            ST_IsValid(geom) AS is_valid_duckdb,
            ST_GeometryType(geom) AS geom_type_duckdb,
            ST_Area(geom) AS area_deg2
        FROM read_parquet('{geoparquet}')
        WHERE {conditions}
    """).df()
    con.close()
    return df


def diagnose_geometry(wkb_bytes: bytes) -> tuple[str, str]:
    """Try to parse and validate a WKB geometry, return (status, detail)."""
    from shapely import from_wkb
    from shapely.validation import explain_validity

    # Stage 1: Can it be parsed?
    try:
        geom = from_wkb(wkb_bytes)
    except Exception as e:
        return "wkb_parse_error", str(e)

    # Stage 2: Is it empty?
    if geom.is_empty:
        return "empty_geometry", f"Parsed as empty {geom.geom_type}"

    # Stage 3: Is it valid?
    if not geom.is_valid:
        reason = explain_validity(geom)
        # Stage 4: Can buffer(0) repair it?
        try:
            repaired = geom.buffer(0)
        except Exception as e:
            return "invalid_unrepairable", f"{reason}; buffer(0) failed: {e}"

        if repaired.is_empty:
            return "degenerate_after_repair", f"{reason}; buffer(0) collapsed to empty"

        if repaired.geom_type not in ("Polygon", "MultiPolygon"):
            return "degenerate_after_repair", f"{reason}; buffer(0) produced {repaired.geom_type}"

        return "invalid_but_repairable", reason

    # Stage 5: Valid geometry — check if it's a polygon type
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        return "wrong_geom_type", f"Type is {geom.geom_type}, expected Polygon/MultiPolygon"

    # Stage 6: Degenerate polygon (zero area)?
    if geom.area == 0:
        return "zero_area", "Valid polygon with zero area"

    return "ok", "No issue detected"


def main():
    parser = argparse.ArgumentParser(description="Audit dropped polygons from a pipeline run")
    parser.add_argument("--request", required=True, help="Input request CSV (project_id, plantstart_year)")
    parser.add_argument("--stats", required=True, help="Output stats CSV from the pipeline run")
    parser.add_argument("--geoparquet", default=GEOPARQUET, help="Path to tm.geoparquet")
    parser.add_argument("-o", "--output", default=None, help="Output CSV for dropped polygon report")
    args = parser.parse_args()

    # Load expected vs actual
    logger.info("Loading expected polygons from request CSV + geoparquet...")
    expected = load_expected_polygons(args.request, args.geoparquet)
    logger.info(f"Expected: {len(expected)} polygons")

    stats = pd.read_csv(args.stats)
    produced_ids = set(stats["poly_uuid"].astype(str))
    logger.info(f"Produced: {len(produced_ids)} polygons in stats output")

    # Find missing
    expected["poly_uuid_str"] = expected["poly_uuid"].astype(str)
    missing = expected[~expected["poly_uuid_str"].isin(produced_ids)].copy()
    logger.info(f"Dropped:  {len(missing)} polygons")

    if missing.empty:
        print("\nAll polygons accounted for — nothing was dropped.")
        return

    # Diagnose each missing polygon
    print(f"\n{'poly_uuid':<40} {'project_id':<40} {'year':>4}  {'status':<28} detail")
    print(f"{'-'*40} {'-'*40} {'-'*4}  {'-'*28} {'-'*50}")

    rows = []
    for _, row in missing.iterrows():
        wkb = row["__wkb"]
        status, detail = diagnose_geometry(bytes(wkb))
        print(f"{row['poly_uuid']:<40} {row['project_id']:<40} {row['plantstart_year']:>4}  {status:<28} {detail}")
        rows.append({
            "poly_uuid": row["poly_uuid"],
            "project_id": row["project_id"],
            "short_name": row["short_name"],
            "plantstart_year": row["plantstart_year"],
            "is_valid_duckdb": row["is_valid_duckdb"],
            "geom_type_duckdb": row["geom_type_duckdb"],
            "area_deg2": row["area_deg2"],
            "status": status,
            "detail": detail,
        })

    # Summary
    status_counts = pd.Series([r["status"] for r in rows]).value_counts()
    print(f"\nSummary:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    if args.output:
        pd.DataFrame(rows).to_csv(args.output, index=False)
        logger.info(f"Report written to {args.output}")
    else:
        default_out = "temp/dropped_polygons_report.csv"
        pd.DataFrame(rows).to_csv(default_out, index=False)
        logger.info(f"Report written to {default_out}")


if __name__ == "__main__":
    main()
