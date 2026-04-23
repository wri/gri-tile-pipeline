"""Identify polygons dropped during a pipeline run and diagnose why.

Compares the set of polygons that *should* have been processed (from the
request CSV + geoparquet) against the output stats CSV to find missing
poly_uuids, then classifies the geometry issue for each.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from gri_tile_pipeline.duckdb_utils import connect_with_spatial


@dataclass
class DropReport:
    n_expected: int
    n_produced: int
    n_dropped: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    status_counts: dict[str, int] = field(default_factory=dict)


def load_expected_polygons(request_csv: str, geoparquet: str) -> pd.DataFrame:
    """Load all poly_uuids that match the request CSV from the geoparquet."""
    req = pd.read_csv(request_csv)
    required = {"project_id", "plantstart_year"}
    if not required.issubset(req.columns):
        raise ValueError(
            f"Request CSV must have columns {sorted(required)}; got {list(req.columns)}"
        )

    conditions = " OR ".join(
        f"(project_id = '{row.project_id}' "
        f"AND YEAR(plantstart) = {int(row.plantstart_year)})"
        for _, row in req.iterrows()
    )

    con = connect_with_spatial()
    try:
        return con.sql(f"""
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
    finally:
        con.close()


def diagnose_geometry(wkb_bytes: bytes) -> tuple[str, str]:
    """Classify a WKB geometry into (status, detail).

    Possible statuses: ok, wkb_parse_error, empty_geometry,
    invalid_but_repairable, invalid_unrepairable, degenerate_after_repair,
    wrong_geom_type, zero_area.
    """
    from shapely import from_wkb
    from shapely.validation import explain_validity

    try:
        geom = from_wkb(wkb_bytes)
    except Exception as exc:
        return "wkb_parse_error", str(exc)

    if geom.is_empty:
        return "empty_geometry", f"Parsed as empty {geom.geom_type}"

    if not geom.is_valid:
        reason = explain_validity(geom)
        try:
            repaired = geom.buffer(0)
        except Exception as exc:
            return "invalid_unrepairable", f"{reason}; buffer(0) failed: {exc}"
        if repaired.is_empty:
            return "degenerate_after_repair", f"{reason}; buffer(0) collapsed to empty"
        if repaired.geom_type not in ("Polygon", "MultiPolygon"):
            return ("degenerate_after_repair",
                    f"{reason}; buffer(0) produced {repaired.geom_type}")
        return "invalid_but_repairable", reason

    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        return "wrong_geom_type", f"Type is {geom.geom_type}, expected Polygon/MultiPolygon"

    if geom.area == 0:
        return "zero_area", "Valid polygon with zero area"

    return "ok", "No issue detected"


def audit_drops(request_csv: str, stats_csv: str, geoparquet: str) -> DropReport:
    """Find polygons that were in the request but missing from the stats output,
    classifying each by geometry status."""
    expected = load_expected_polygons(request_csv, geoparquet)
    stats = pd.read_csv(stats_csv)
    produced_ids = set(stats["poly_uuid"].astype(str))

    expected["poly_uuid_str"] = expected["poly_uuid"].astype(str)
    missing = expected[~expected["poly_uuid_str"].isin(produced_ids)].copy()

    rows = []
    for _, row in missing.iterrows():
        status, detail = diagnose_geometry(bytes(row["__wkb"]))
        rows.append({
            "poly_uuid": row["poly_uuid"],
            "project_id": row["project_id"],
            "short_name": row["short_name"],
            "plantstart_year": row["plantstart_year"],
            "is_valid_duckdb": bool(row["is_valid_duckdb"]),
            "geom_type_duckdb": row["geom_type_duckdb"],
            "area_deg2": float(row["area_deg2"]),
            "status": status,
            "detail": detail,
        })

    status_counts = pd.Series([r["status"] for r in rows]).value_counts().to_dict() if rows else {}

    return DropReport(
        n_expected=len(expected),
        n_produced=len(produced_ids),
        n_dropped=len(missing),
        rows=rows,
        status_counts=status_counts,
    )
