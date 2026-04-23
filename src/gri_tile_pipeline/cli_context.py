"""Shared CLI context stashed on click.Context.obj.

Every subcommand reads the user's cross-cutting preferences (verbose, quiet,
json, workers, dry-run, yes, aws-profile, run-history-dir) from here instead
of redeclaring flags. Subcommands may override individual values with local
flags when the user passes both.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import click

from gri_tile_pipeline.config import PipelineConfig


CTX_KEY = "gri_ctx"


def resolve_pipeline_version() -> str:
    """Return installed ``gri-tile-pipeline`` version, or ``"unknown"``."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("gri-tile-pipeline")
    except Exception:
        return "unknown"


def resolve_git_sha() -> Optional[str]:
    """Return the pipeline's git commit sha, or None if unavailable.

    Precedence:
      1. ``GRI_GIT_SHA`` env var (dev overrides, CI injection).
      2. ``docker/.git_sha`` / ``/function/.git_sha`` — written by
         ``infra/Makefile`` before the predict image build and copied
         into the Lambda image (see ``docker/PredictDockerfile``).
      3. ``git rev-parse HEAD`` for local developer runs.
    """
    env_sha = (os.environ.get("GRI_GIT_SHA") or "").strip()
    if env_sha:
        return env_sha
    for path in ("/function/.git_sha", "/var/task/.git_sha", "docker/.git_sha"):
        try:
            with open(path) as f:
                sha = f.read().strip()
                if sha and sha != "unknown":
                    return sha
        except OSError:
            continue
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            return sha or None
    except Exception:
        pass
    return None


@dataclass
class CliContext:
    cfg: PipelineConfig
    run_id: str
    verbose: int = 0          # -v count; 0 = INFO, 1 = DEBUG, 2+ = TRACE
    quiet: bool = False       # promotes log level to WARNING; mutually exclusive with verbose
    json_mode: bool = False   # emit JSON on stdout; text logs go to stderr only
    workers: int = 1          # local-mode parallelism
    dry_run: bool = False
    yes: bool = False         # skip confirmation prompts
    aws_profile: Optional[str] = None
    run_history_dir: str = "runs"
    pipeline_version: str = "unknown"
    git_sha: Optional[str] = None


def attach(click_ctx: click.Context, gri_ctx: CliContext) -> None:
    """Store the CliContext on click's ctx.obj under a stable key."""
    click_ctx.ensure_object(dict)
    click_ctx.obj[CTX_KEY] = gri_ctx
    # Mirror common fields at the top level so legacy ctx.obj["cfg"] / ["run_id"] keep working.
    click_ctx.obj["cfg"] = gri_ctx.cfg
    click_ctx.obj["run_id"] = gri_ctx.run_id


def get(click_ctx: click.Context) -> CliContext:
    """Pull the CliContext off any click.Context in the tree."""
    # Walk up the parent chain in case a subcommand was invoked directly.
    ctx: Optional[click.Context] = click_ctx
    while ctx is not None:
        if ctx.obj and CTX_KEY in ctx.obj:
            return ctx.obj[CTX_KEY]
        ctx = ctx.parent
    raise RuntimeError("CliContext not attached; root group did not run.")


def emit_json(click_ctx: click.Context, payload: dict[str, Any]) -> None:
    """Emit a single JSON document on stdout when --json is active.

    No-op when json_mode is off — callers should also print/log text output
    in that case.
    """
    gri = get(click_ctx)
    if not gri.json_mode:
        return
    json.dump(payload, sys.stdout, default=str, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


def confirm_or_abort(click_ctx: click.Context, message: str) -> None:
    """Interactive yes/no gate that respects --yes, --dry-run, and --json.

    --yes  -> auto-accept
    --dry-run -> never execute; caller should have bailed earlier, but we accept
                 so dry-run paths can still reach the no-op code paths.
    --json -> auto-accept; prompts don't belong in machine-readable mode.
    """
    gri = get(click_ctx)
    if gri.yes or gri.dry_run or gri.json_mode:
        return
    click.confirm(message, abort=True)
