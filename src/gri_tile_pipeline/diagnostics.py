"""``gri-ttc doctor``: verify a workstation is ready to run the pipeline.

Each check is independent and returns a :class:`CheckResult`. The `doctor`
command composes them and reports pass/fail for each. No check raises; all
failures are captured in the result payload so the full report always renders.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gri_tile_pipeline.config import LITHOPS_ENV_VAR, PipelineConfig


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    hint: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "hint": self.hint,
        }


def check_config_parse(config_path: Optional[str] = None) -> CheckResult:
    """Verify the config YAML parses (or that defaults are usable)."""
    from gri_tile_pipeline.config import load_config

    try:
        load_config(config_path)
    except Exception as e:
        return CheckResult(
            "config", ok=False,
            detail=f"failed to load config: {e}",
            hint="copy config_template.yaml -> config.yaml and fix the invalid keys",
        )
    target = config_path or "config.yaml"
    return CheckResult("config", ok=True, detail=f"loaded {target}")


def check_secrets_parse(secrets_path: str = "secrets.yaml") -> CheckResult:
    """Verify secrets.yaml parses if present; silent if missing."""
    p = Path(secrets_path)
    if not p.exists():
        return CheckResult(
            "secrets", ok=True,
            detail=f"{secrets_path} not present (ok; using env vars)",
        )
    try:
        import yaml

        with open(p) as f:
            yaml.safe_load(f)
    except Exception as e:
        return CheckResult(
            "secrets", ok=False,
            detail=f"failed to parse {secrets_path}: {e}",
            hint="check YAML syntax, copy secrets_template.yaml if needed",
        )
    return CheckResult("secrets", ok=True, detail=f"parsed {secrets_path}")


def check_aws_credentials(profile: Optional[str] = None) -> CheckResult:
    """Confirm boto3 can resolve AWS credentials in this environment."""
    try:
        import boto3
    except ImportError:
        return CheckResult(
            "aws-credentials", ok=False,
            detail="boto3 not installed",
            hint="pip install 'gri-tile-pipeline[loaders]' (adds boto3)",
        )
    try:
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        creds = session.get_credentials()
    except Exception as e:
        return CheckResult(
            "aws-credentials", ok=False,
            detail=f"boto3 raised: {e}",
            hint="configure AWS creds (aws configure sso / AWS_PROFILE / env vars)",
        )
    if creds is None:
        return CheckResult(
            "aws-credentials", ok=False,
            detail="no AWS credentials resolvable",
            hint="run `aws configure sso` or export AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY",
        )
    label = profile or os.environ.get("AWS_PROFILE") or "default"
    return CheckResult(
        "aws-credentials", ok=True, detail=f"resolved via profile={label}",
    )


def check_lithops_env() -> CheckResult:
    """Verify the ``LITHOPS_ENV`` selection points at rendered config files."""
    env = os.environ.get(LITHOPS_ENV_VAR)
    if not env:
        return CheckResult(
            "lithops-env", ok=True,
            detail=f"{LITHOPS_ENV_VAR} not set (ok if you stay in --local mode)",
            hint=f"export {LITHOPS_ENV_VAR}=land-research to run on Lambda",
        )
    base = Path(f".lithops/{env}")
    expected = [
        base / "config.loaders-euc1.yaml",
        base / "config.loaders-usw2.yaml",
        base / "config.s1.yaml",
        base / "config.predict.yaml",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        return CheckResult(
            "lithops-env", ok=False,
            detail=f"{LITHOPS_ENV_VAR}={env} but missing: {', '.join(missing)}",
            hint="run `make render ENV=<env>` from infra/ to generate these",
        )
    return CheckResult(
        "lithops-env", ok=True,
        detail=f"{LITHOPS_ENV_VAR}={env} -> {base}/* present",
    )


def check_geoparquet(
    cfg: PipelineConfig,
    local_path: str = "temp/tm.geoparquet",
    remote_uri: str = "s3://wri-restoration-geodata/terramatch/tm.geoparquet",
    profile: Optional[str] = None,
) -> CheckResult:
    """Check geoparquet reachability: local first, then remote HEAD."""
    lp = Path(local_path)
    if lp.exists():
        mtime = lp.stat().st_mtime
        from datetime import datetime, timezone

        age_days = (
            datetime.now(tz=timezone.utc)
            - datetime.fromtimestamp(mtime, tz=timezone.utc)
        ).days
        hint = (
            "geoparquet is older than 14 days; pull a fresh copy"
            if age_days > 14 else None
        )
        return CheckResult(
            "geoparquet", ok=True,
            detail=f"local {local_path} ({age_days}d old)",
            hint=hint,
        )

    try:
        import boto3  # noqa: F401
    except ImportError:
        return CheckResult(
            "geoparquet", ok=False,
            detail=f"{local_path} missing and boto3 not installed",
            hint=f"download to {local_path} or `pip install boto3`",
        )

    try:
        import boto3

        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        s3 = session.client("s3")
        _, _, rest = remote_uri.partition("s3://")
        bucket, _, key = rest.partition("/")
        s3.head_object(Bucket=bucket, Key=key)
    except Exception as e:
        return CheckResult(
            "geoparquet", ok=False,
            detail=f"HEAD {remote_uri} failed: {e}",
            hint=(
                f"check read access to {remote_uri} or download to {local_path}"
            ),
        )
    return CheckResult(
        "geoparquet", ok=True,
        detail=f"{remote_uri} reachable (no local cache at {local_path})",
        hint=f"for fast repeated runs, copy to {local_path}",
    )


def check_terramatch(env: str = "staging") -> CheckResult:
    """Probe the TerraMatch API with a 1-item page size to validate the token."""
    from gri_tile_pipeline.terramatch.client import TMApiError, TMClient
    from gri_tile_pipeline.terramatch.secrets import (
        MissingTMCredential,
        resolve_tm_creds,
    )

    try:
        base_url, token = resolve_tm_creds(env)  # type: ignore[arg-type]
    except MissingTMCredential as e:
        return CheckResult(
            "terramatch", ok=False,
            detail=str(e),
            hint="skip with --skip-tm if TerraMatch patching is not needed",
        )

    client = TMClient(base_url, token, timeout=10.0)
    try:
        client._get("/sitePolygons", params={"page[size]": 1})
    except TMApiError as e:
        return CheckResult(
            "terramatch", ok=False,
            detail=f"{env} API returned {e.status}",
            hint="confirm the token is valid for the target environment",
        )
    except Exception as e:
        return CheckResult(
            "terramatch", ok=False,
            detail=f"network/transport error: {e}",
            hint="check connectivity to the TerraMatch API",
        )
    return CheckResult(
        "terramatch", ok=True, detail=f"{env} API reachable ({base_url})",
    )


def run_checks(
    *,
    cfg: PipelineConfig,
    config_path: Optional[str] = None,
    aws_profile: Optional[str] = None,
    check_tm: bool = False,
    tm_env: str = "staging",
) -> list[CheckResult]:
    results = [
        check_config_parse(config_path),
        check_secrets_parse(),
        check_aws_credentials(aws_profile),
        check_lithops_env(),
        check_geoparquet(cfg, profile=aws_profile),
    ]
    if check_tm:
        results.append(check_terramatch(tm_env))
    return results
