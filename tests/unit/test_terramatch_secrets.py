"""Tests for TerraMatch credential resolution."""

from __future__ import annotations

import pytest
import yaml

from gri_tile_pipeline.terramatch.secrets import (
    DEFAULT_SECRETS_PATH,
    MissingTMCredential,
    resolve_tm_creds,
)
from gri_tile_pipeline.terramatch.client import (
    DEFAULT_PRODUCTION_URL,
    DEFAULT_STAGING_URL,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("GRI_TM_TOKEN", raising=False)


def test_resolves_token_from_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    url, tok = resolve_tm_creds("staging", token="flag-token")
    assert tok == "flag-token"
    assert url == DEFAULT_STAGING_URL


def test_env_var_beats_secrets_file(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text(yaml.safe_dump({"terramatch": {"token": "file-token"}}))
    monkeypatch.setenv("GRI_TM_TOKEN", "env-token")
    url, tok = resolve_tm_creds("staging", secrets_path=secrets)
    assert tok == "env-token"


def test_reads_secrets_file_when_no_env(tmp_path):
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text(yaml.safe_dump({
        "terramatch": {
            "token": "file-token",
            "production_url": "https://custom.example/api",
        }
    }))
    url, tok = resolve_tm_creds("production", secrets_path=secrets)
    assert tok == "file-token"
    assert url == "https://custom.example/api"


def test_production_env_default_url(tmp_path, monkeypatch):
    monkeypatch.setenv("GRI_TM_TOKEN", "t")
    url, _ = resolve_tm_creds("production", secrets_path=tmp_path / "nope.yaml")
    assert url == DEFAULT_PRODUCTION_URL


def test_base_url_override_wins(tmp_path):
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text(yaml.safe_dump({
        "terramatch": {"token": "t", "staging_url": "https://x.example/api"}
    }))
    url, _ = resolve_tm_creds(
        "staging", base_url="https://cli.example/api", secrets_path=secrets,
    )
    assert url == "https://cli.example/api"


def test_missing_token_raises(tmp_path):
    with pytest.raises(MissingTMCredential):
        resolve_tm_creds("staging", secrets_path=tmp_path / "nope.yaml")


def test_default_secrets_path_is_relative():
    assert DEFAULT_SECRETS_PATH == "secrets.yaml"
