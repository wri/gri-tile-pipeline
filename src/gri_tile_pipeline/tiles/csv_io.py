"""Read / write the standard tiles CSV format used by both orchestrators."""

from __future__ import annotations

import csv
from typing import Any, Dict, List


REQUIRED_COLUMNS = {"Year", "X", "Y", "Y_tile", "X_tile"}


def read_tiles_csv(path: str) -> List[Dict[str, Any]]:
    """Read a tiles CSV into a list of dicts.

    Expected columns: ``Year, X, Y, Y_tile, X_tile``
    where *X* = lon (float) and *Y* = lat (float).
    """
    rows: List[Dict[str, Any]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        for row in reader:
            rows.append({
                "year": int(row["Year"]),
                "lon": float(row["X"]),
                "lat": float(row["Y"]),
                "X_tile": int(row["X_tile"]),
                "Y_tile": int(row["Y_tile"]),
            })
    return rows


def write_tiles_csv(path: str, tiles: List[Dict[str, Any]]) -> None:
    """Write a list of tile dicts back to the standard CSV format."""
    fieldnames = ["Year", "X", "Y", "Y_tile", "X_tile"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in tiles:
            writer.writerow({
                "Year": t["year"],
                "X": t["lon"],
                "Y": t["lat"],
                "Y_tile": t["Y_tile"],
                "X_tile": t["X_tile"],
            })
