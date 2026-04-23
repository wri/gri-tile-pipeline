"""Resolve TerraMatch credentials from CLI flags, env vars, and ``secrets.yaml``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml

from gri_tile_pipeline.terramatch.client import (
    DEFAULT_PRODUCTION_URL,
    DEFAULT_STAGING_URL,
)

Env = Literal["staging", "production"]

TOKEN_ENV_VAR = "GRI_TM_TOKEN"
DEFAULT_SECRETS_PATH = "secrets.yaml"


class MissingTMCredential(RuntimeError):
    """Raised when a required TerraMatch credential can't be resolved."""


def load_secrets(path: str | Path = DEFAULT_SECRETS_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def resolve_tm_creds(
    env: Env,
    *,
    token: Optional[str] = None,
    base_url: Optional[str] = None,
    secrets_path: str | Path = DEFAULT_SECRETS_PATH,
) -> tuple[str, str]:
    """Return ``(base_url, token)`` for the requested environment.

    Precedence:
      - token: ``--token`` arg > ``GRI_TM_TOKEN`` env var > ``secrets.yaml:terramatch.token``
      - base_url: ``--base-url`` arg > ``secrets.yaml:terramatch.<env>_url`` > built-in default
    """
    secrets = load_secrets(secrets_path)
    tm = secrets.get("terramatch") or {}

    resolved_token = token or os.environ.get(TOKEN_ENV_VAR) or tm.get("token")
    if not resolved_token:
        raise MissingTMCredential(
            f"TerraMatch token not found. Set {TOKEN_ENV_VAR} env var, pass --token, "
            f"or add `terramatch.token` to {secrets_path}."
        )

    default_urls = {
        "staging": DEFAULT_STAGING_URL,
        "production": DEFAULT_PRODUCTION_URL,
    }
    resolved_url = base_url or tm.get(f"{env}_url") or default_urls[env]

    return str(resolved_url).rstrip("/"), str(resolved_token)
