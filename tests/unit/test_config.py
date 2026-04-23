"""Unit tests for config.py."""

import tempfile
import os

import pytest

from gri_tile_pipeline.config import (
    DownloadConfig,
    LithopsConfig,
    PipelineConfig,
    PredictConfig,
    S1RTCConfig,
    ZonalConfig,
    load_config,
)


def test_default_values():
    cfg = PipelineConfig()
    assert cfg.parquet_path == "data/tiledb.parquet"
    assert cfg.download.memory_mb == 4096
    assert cfg.predict.memory_mb == 6144
    assert cfg.predict.timeout_sec == 600
    assert cfg.zonal.tile_bucket == "tof-output"


def test_load_config_nonexistent_returns_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(None)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.download.retries == 3


def test_load_config_from_yaml(tmp_path):
    yaml_content = """\
pipeline:
  parquet_path: /custom/path.parquet
download:
  memory_mb: 2048
  retries: 5
predict:
  timeout_sec: 1200
"""
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(yaml_content)

    cfg = load_config(str(cfg_path))
    assert cfg.parquet_path == "/custom/path.parquet"
    assert cfg.download.memory_mb == 2048
    assert cfg.download.retries == 5
    assert cfg.predict.timeout_sec == 1200
    # Unchanged defaults
    assert cfg.predict.memory_mb == 6144


def test_partial_yaml_override(tmp_path):
    yaml_content = """\
zonal:
  tile_bucket: my-bucket
"""
    cfg_path = tmp_path / "partial.yaml"
    cfg_path.write_text(yaml_content)

    cfg = load_config(str(cfg_path))
    assert cfg.zonal.tile_bucket == "my-bucket"
    assert cfg.zonal.small_sites_area_thresh == 0.5  # default preserved


def test_empty_yaml(tmp_path):
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("")

    cfg = load_config(str(cfg_path))
    assert isinstance(cfg, PipelineConfig)


def test_real_config_if_exists():
    """Load the real config.yaml if it exists in the repo root."""
    real_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml"
    )
    if os.path.isfile(real_config):
        cfg = load_config(real_config)
        assert isinstance(cfg, PipelineConfig)


def test_lithops_env_retargets_paths(tmp_path, monkeypatch):
    """LITHOPS_ENV=foo should retarget all four lithops paths to .lithops/foo/."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LITHOPS_ENV", "land-research")
    cfg = load_config(None)
    assert cfg.lithops.euc1_config == ".lithops/land-research/config.loaders-euc1.yaml"
    assert cfg.lithops.usw2_config == ".lithops/land-research/config.loaders-usw2.yaml"
    assert cfg.lithops.s1_usw2_config == ".lithops/land-research/config.s1.yaml"
    assert cfg.lithops.predict_config == ".lithops/land-research/config.predict.yaml"


def test_lithops_env_unset_yields_empty_predict_config(tmp_path, monkeypatch):
    """With LITHOPS_ENV unset, predict_config defaults to empty string.

    The old legacy us-west-2 path was removed to prevent silent cross-region
    deploys. The predict step itself raises if this is still empty at fan-out.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LITHOPS_ENV", raising=False)
    cfg = load_config(None)
    assert cfg.lithops.predict_config == ""


def test_yaml_overrides_lithops_env(tmp_path, monkeypatch):
    """Per-key YAML override wins over LITHOPS_ENV."""
    monkeypatch.setenv("LITHOPS_ENV", "land-research")
    yaml_content = """\
lithops:
  predict_config: /explicit/predict.yaml
"""
    cfg_path = tmp_path / "explicit.yaml"
    cfg_path.write_text(yaml_content)
    cfg = load_config(str(cfg_path))
    assert cfg.lithops.predict_config == "/explicit/predict.yaml"
    # Non-overridden keys still come from LITHOPS_ENV:
    assert cfg.lithops.usw2_config == ".lithops/land-research/config.loaders-usw2.yaml"
