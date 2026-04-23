"""Tests for the `gri-ttc doctor` checks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from gri_tile_pipeline.config import PipelineConfig
from gri_tile_pipeline.diagnostics import (
    check_config_parse,
    check_geoparquet,
    check_lithops_env,
    check_secrets_parse,
    check_terramatch,
    run_checks,
)


def test_check_config_parse_with_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = check_config_parse(None)
    assert r.ok is True


def test_check_config_parse_bad_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(":::not valid yaml:\n  - [")
    r = check_config_parse(str(bad))
    assert r.ok is False
    assert "failed to load" in r.detail


def test_check_secrets_parse_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = check_secrets_parse("secrets.yaml")
    assert r.ok is True
    assert "not present" in r.detail


def test_check_secrets_parse_valid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("secrets.yaml").write_text(yaml.safe_dump({"terramatch": {"token": "x"}}))
    r = check_secrets_parse("secrets.yaml")
    assert r.ok is True


def test_check_secrets_parse_broken_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Genuinely malformed: unclosed flow mapping + bad indentation.
    Path("secrets.yaml").write_text(
        "terramatch: {token: 'abc'\nother_key:\n\t- not allowed\n"
    )
    r = check_secrets_parse("secrets.yaml")
    assert r.ok is False
    assert r.hint


def test_check_lithops_env_not_set(monkeypatch):
    monkeypatch.delenv("LITHOPS_ENV", raising=False)
    r = check_lithops_env()
    assert r.ok is True
    assert "not set" in r.detail


def test_check_lithops_env_missing_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LITHOPS_ENV", "nope")
    r = check_lithops_env()
    assert r.ok is False
    assert "config.loaders-euc1.yaml" in r.detail


def test_check_lithops_env_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LITHOPS_ENV", "dev")
    base = tmp_path / ".lithops" / "dev"
    base.mkdir(parents=True)
    for name in ("config.loaders-euc1.yaml", "config.loaders-usw2.yaml",
                 "config.s1.yaml", "config.predict.yaml"):
        (base / name).write_text("dummy")
    r = check_lithops_env()
    assert r.ok is True


def test_check_geoparquet_local_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "temp" / "tm.geoparquet"
    local.parent.mkdir()
    local.write_bytes(b"fake")
    r = check_geoparquet(PipelineConfig(), local_path=str(local))
    assert r.ok is True
    assert "local" in r.detail


def test_check_terramatch_missing_creds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GRI_TM_TOKEN", raising=False)
    r = check_terramatch("staging")
    assert r.ok is False
    assert "token" in r.detail.lower()


def test_check_terramatch_api_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRI_TM_TOKEN", "bad")
    from gri_tile_pipeline.terramatch.client import TMApiError

    with patch.object(
        __import__("gri_tile_pipeline.terramatch.client", fromlist=["TMClient"]).TMClient,
        "_get", side_effect=TMApiError(401, "unauth"),
    ):
        r = check_terramatch("staging")
    assert r.ok is False
    assert "401" in r.detail


def test_run_checks_aggregates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LITHOPS_ENV", raising=False)
    # Temp geoparquet so this check passes; we don't want the test hitting S3.
    gp = tmp_path / "temp" / "tm.geoparquet"
    gp.parent.mkdir()
    gp.write_bytes(b"x")
    with patch("gri_tile_pipeline.diagnostics.check_aws_credentials") as aws_chk:
        aws_chk.return_value.ok = True
        aws_chk.return_value.name = "aws-credentials"
        aws_chk.return_value.detail = "stubbed"
        aws_chk.return_value.hint = None
        results = run_checks(cfg=PipelineConfig())
    names = [r.name for r in results]
    assert "config" in names
    assert "secrets" in names
    assert "aws-credentials" in names
    assert "lithops-env" in names
    assert "geoparquet" in names
    assert "terramatch" not in names  # not requested
