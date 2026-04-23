"""Orchestrate patching TTC stats back onto TerraMatch polygons.

Input is a ``results.csv`` emitted by ``gri-ttc stats`` (one row per polygon,
with at minimum a ``poly_uuid`` column and a percent-cover column).

Matches each row's ``poly_uuid`` against the ids returned by
``GET /sitePolygons`` for the project, then PATCHes one indicator per row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional

from loguru import logger

from gri_tile_pipeline.terramatch.client import TMApiError, TMClient

DEFAULT_SLUG = "treeCover"
DEFAULT_PROJECT_PHASE = "implementation"
DEFAULT_PERCENT_COLUMN = "ttc"
DEFAULT_YEAR_COLUMN = "year"

# Ordered candidates — first match wins. Columns produced by the zonal error-
# propagation step when `--shift-error` is on live here; `_error` / `_stderr` /
# `_uncertainty` suffixes are accepted as a fallback.
UNCERTAINTY_COLUMN_CANDIDATES: tuple[str, ...] = (
    "ttc_shift_error",
    "ttc_error",
    "ttc_stderr",
    "ttc_uncertainty",
    "shift_error",
    "plus_minus_percent",
)


@dataclass
class IndicatorSpec:
    year: Optional[int] = None  # required if no year_column value is set on the row
    slug: str = DEFAULT_SLUG
    project_phase: str = DEFAULT_PROJECT_PHASE
    year_column: Optional[str] = DEFAULT_YEAR_COLUMN
    percent_column: str = DEFAULT_PERCENT_COLUMN
    uncertainty_column: Optional[str] = None


PatchStatus = Literal["sent", "dryrun", "unmatched", "error"]


@dataclass
class PatchOutcome:
    poly_uuid: str
    polygon_id: Optional[str]
    status: PatchStatus
    http_status: Optional[int] = None
    message: Optional[str] = None
    payload: Optional[dict] = field(default=None, repr=False)

    def as_dict(self) -> dict:
        out = {
            "poly_uuid": self.poly_uuid,
            "polygon_id": self.polygon_id,
            "status": self.status,
        }
        if self.http_status is not None:
            out["http_status"] = self.http_status
        if self.message:
            out["message"] = self.message
        return out


def load_results(csv_path: str | Path) -> tuple[list[dict], list[str]]:
    """Return (rows, column_names) from *csv_path*."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records"), list(df.columns)


def detect_uncertainty_column(columns: Iterable[str]) -> Optional[str]:
    """Find the shift-error column emitted by ``stats --shift-error``, if any."""
    cols = list(columns)
    lower_to_actual = {c.lower(): c for c in cols}
    for cand in UNCERTAINTY_COLUMN_CANDIDATES:
        if cand in lower_to_actual:
            return lower_to_actual[cand]
    for col in cols:
        low = col.lower()
        if low.endswith(("_error", "_stderr", "_uncertainty")):
            return col
    return None


def build_poly_id_set(client: TMClient, project_id: str) -> set[str]:
    """Fetch every polygon id TerraMatch knows about for *project_id*."""
    ids: set[str] = set()
    for item in client.list_site_polygons(project_id):
        pid = item.get("id")
        if pid:
            ids.add(str(pid))
    return ids


def _coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def build_indicator(row: dict, spec: IndicatorSpec) -> dict:
    """Assemble one indicator dict for the PATCH payload.

    Raises :class:`ValueError` if the percent-cover column is missing or non-numeric.
    """
    year: Optional[int] = spec.year
    if spec.year_column and spec.year_column in row:
        raw = row[spec.year_column]
        if raw not in (None, ""):
            try:
                if not (isinstance(raw, float) and math.isnan(raw)):
                    year = int(raw)
            except (TypeError, ValueError):
                pass
    if year is None:
        raise ValueError(
            f"no year available (pass --year or include {spec.year_column!r} column)"
        )

    percent = _coerce_float(row.get(spec.percent_column))
    if percent is None:
        raise ValueError(
            f"row missing numeric {spec.percent_column!r} column"
        )

    indicator: dict = {
        "indicatorSlug": spec.slug,
        "yearOfAnalysis": int(year),
        "projectPhase": spec.project_phase,
        "percentCover": percent,
    }

    if spec.uncertainty_column:
        unc = _coerce_float(row.get(spec.uncertainty_column))
        if unc is not None:
            indicator["plusMinusPercent"] = unc

    return indicator


def run_patch(
    rows: list[dict],
    project_id: str,
    client: TMClient,
    spec: IndicatorSpec,
    *,
    apply: bool,
    limit: Optional[int] = None,
) -> list[PatchOutcome]:
    """Patch every eligible *row* onto TerraMatch. Never raises for single-row failures."""
    try:
        known_ids = build_poly_id_set(client, project_id)
    except TMApiError as e:
        raise RuntimeError(
            f"Failed to list polygons for project {project_id!r}: {e}"
        ) from e

    logger.info(
        f"TerraMatch project {project_id}: "
        f"{len(known_ids)} polygon(s) returned by /sitePolygons"
    )

    to_process = rows[:limit] if limit else rows
    outcomes: list[PatchOutcome] = []
    for row in to_process:
        raw_uuid = row.get("poly_uuid")
        poly_uuid = "" if raw_uuid is None else str(raw_uuid)
        if not poly_uuid:
            outcomes.append(PatchOutcome(
                poly_uuid="", polygon_id=None, status="error",
                message="row has no poly_uuid",
            ))
            continue
        if poly_uuid not in known_ids:
            outcomes.append(PatchOutcome(
                poly_uuid=poly_uuid, polygon_id=None, status="unmatched",
                message="not returned by /sitePolygons for this project",
            ))
            continue
        try:
            indicator = build_indicator(row, spec)
        except ValueError as e:
            outcomes.append(PatchOutcome(
                poly_uuid=poly_uuid, polygon_id=poly_uuid,
                status="error", message=str(e),
            ))
            continue

        if not apply:
            logger.info(f"[dry-run] {poly_uuid}: would PATCH {indicator}")
            outcomes.append(PatchOutcome(
                poly_uuid=poly_uuid, polygon_id=poly_uuid,
                status="dryrun", payload={"indicators": [indicator]},
            ))
            continue

        try:
            resp = client.patch_site_polygon(poly_uuid, [indicator])
            logger.info(f"PATCH {poly_uuid} -> {resp.status_code}")
            outcomes.append(PatchOutcome(
                poly_uuid=poly_uuid, polygon_id=poly_uuid,
                status="sent", http_status=resp.status_code,
            ))
        except TMApiError as e:
            logger.error(f"PATCH {poly_uuid} failed: {e}")
            outcomes.append(PatchOutcome(
                poly_uuid=poly_uuid, polygon_id=poly_uuid,
                status="error", http_status=e.status,
                message=e.body[:500] if e.body else str(e),
            ))
        except Exception as e:  # network, timeout, etc.
            logger.error(f"PATCH {poly_uuid} raised: {e}")
            outcomes.append(PatchOutcome(
                poly_uuid=poly_uuid, polygon_id=poly_uuid,
                status="error", message=str(e),
            ))

    return outcomes


def summarize(outcomes: list[PatchOutcome]) -> dict:
    counts: dict[str, int] = {"sent": 0, "dryrun": 0, "unmatched": 0, "error": 0}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    counts["total"] = len(outcomes)
    return counts
