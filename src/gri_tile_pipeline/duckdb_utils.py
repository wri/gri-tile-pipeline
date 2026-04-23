"""Shared DuckDB helpers."""

from __future__ import annotations


def connect_with_spatial():
    """Return a DuckDB connection with the spatial extension loaded."""
    import duckdb

    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")
    return con
