"""Centralized logging configuration with run context."""

from __future__ import annotations

import sys
import uuid
from typing import Optional

from loguru import logger


def _level_from_verbose(verbose: int, quiet: bool) -> str:
    if quiet:
        return "WARNING"
    if verbose >= 2:
        return "TRACE"
    if verbose == 1:
        return "DEBUG"
    return "INFO"


def setup_logging(
    level: Optional[str] = None,
    fmt: str = "text",
    verbose: int = 0,
    quiet: bool = False,
    json_mode: bool = False,
) -> None:
    """Configure loguru. Call once at CLI startup.

    Precedence for log level:
        1. Explicit *level* argument (legacy --log-level path)
        2. verbose/quiet counts (new -v/-q flags)

    When *json_mode* is true, logs always go to stderr in serialized form so
    stdout stays clean for the JSON payload each subcommand emits.
    """
    resolved_level = (level or _level_from_verbose(verbose, quiet)).upper()
    logger.remove()

    if json_mode or fmt == "json":
        logger.add(sys.stderr, level=resolved_level, serialize=True)
    else:
        logger.add(
            sys.stderr,
            level=resolved_level,
            format="<level>{level: <8}</level> | {extra[run_id]:>8} | {message}",
        )


def new_run_id() -> str:
    """Generate an 8-char hex run identifier."""
    return uuid.uuid4().hex[:8]


def bind_run_context(run_id: str) -> None:
    """Set *run_id* as a default extra value for all subsequent log calls."""
    logger.configure(extra={"run_id": run_id})
