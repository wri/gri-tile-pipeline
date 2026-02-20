"""Centralized logging configuration with run context."""

from __future__ import annotations

import sys
import uuid

from loguru import logger


def setup_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Configure loguru with the given level and format.

    Call once at CLI startup.  Workers on Lambda that don't call this
    get loguru's default behaviour (no run_id, no harm done).
    """
    logger.remove()
    if fmt == "json":
        logger.add(sys.stderr, level=level.upper(), serialize=True)
    else:
        logger.add(
            sys.stderr,
            level=level.upper(),
            format="<level>{level: <8}</level> | {extra[run_id]:>8} | {message}",
        )


def new_run_id() -> str:
    """Generate an 8-char hex run identifier."""
    return uuid.uuid4().hex[:8]


def bind_run_context(run_id: str) -> None:
    """Set *run_id* as a default extra value for all subsequent log calls."""
    logger.configure(extra={"run_id": run_id})
