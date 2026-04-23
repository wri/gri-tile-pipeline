"""Ingest step: convert inbound JSON request to tiles CSV."""

from __future__ import annotations

import os
from typing import Optional

from loguru import logger

from gri_tile_pipeline.tiles.tile_lookup import resolve_tiles


def run_ingest(
    input_json: str,
    parquet_path: str,
    output: str,
    x_col: str = "X_tile",
    y_col: str = "Y_tile",
    limit: Optional[int] = None,
) -> int:
    """Resolve JSON request into tiles CSV.

    Returns the number of tiles written.
    """
    logger.info(f"Ingesting {input_json} against {parquet_path}")
    result_df = resolve_tiles(
        input_json,
        parquet_path,
        x_col=x_col,
        y_col=y_col,
        limit=limit,
    )

    if result_df.empty:
        logger.warning("No tiles resolved from input JSON")
        os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
        result_df.to_csv(output, index=False)
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    result_df.to_csv(output, index=False)
    logger.info(f"Wrote {len(result_df)} tiles to {output}")
    return len(result_df)
