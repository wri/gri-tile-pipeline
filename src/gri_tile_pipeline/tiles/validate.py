"""Validate a tiles CSV: schema conformance and (optionally) S3 presence."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from gri_tile_pipeline.tiles.csv_io import read_tiles_csv


REQUIRED_COLUMNS = {"Year", "X", "Y", "X_tile", "Y_tile"}


@dataclass
class ValidationReport:
    path: str
    n_rows: int = 0
    schema_ok: bool = False
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    availability: dict | None = None  # populated when check_s3=True

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "n_rows": self.n_rows,
            "schema_ok": self.schema_ok,
            "missing_columns": self.missing_columns,
            "extra_columns": self.extra_columns,
            "parse_errors": self.parse_errors,
            "availability": self.availability,
        }


def validate_tiles_csv(
    path: str,
    *,
    check_s3: bool = False,
    dest: str | None = None,
    region: str = "us-east-1",
    profile: str | None = None,
    check_type: str = "predictions",
) -> ValidationReport:
    """Inspect *path* for schema conformance; optionally ping S3 for each tile."""
    report = ValidationReport(path=path)

    src = Path(path)
    if not src.is_file():
        report.parse_errors.append(f"File not found: {path}")
        return report

    with src.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        report.missing_columns = sorted(REQUIRED_COLUMNS - cols)
        report.extra_columns = sorted(cols - REQUIRED_COLUMNS)
        report.schema_ok = not report.missing_columns

    try:
        tiles = read_tiles_csv(path)
        report.n_rows = len(tiles)
    except Exception as exc:
        report.parse_errors.append(str(exc))
        return report

    if check_s3:
        if not dest:
            report.parse_errors.append("check_s3 requires dest")
            return report
        from gri_tile_pipeline.storage.obstore_utils import from_dest
        from gri_tile_pipeline.tiles.availability import check_availability

        store = from_dest(dest, region=region, profile=profile)
        avail = check_availability(tiles, dest, check_type=check_type, store=store)
        report.availability = {
            "n_existing": len(avail["existing"]),
            "n_missing": len(avail["missing"]),
        }

    return report
