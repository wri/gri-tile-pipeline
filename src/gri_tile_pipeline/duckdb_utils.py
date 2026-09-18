"""Shared DuckDB helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb


def connect_with_spatial() -> "duckdb.DuckDBPyConnection":
    """Return a DuckDB connection with the spatial extension loaded."""
    import duckdb

    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")
    return con
