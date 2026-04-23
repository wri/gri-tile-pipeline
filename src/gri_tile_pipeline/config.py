"""YAML config loading with dataclass defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


LITHOPS_ENV_VAR = "LITHOPS_ENV"


def _lithops_paths_for_env(env: str) -> dict[str, str]:
    """Paths produced by infra/lithops/render.py for a given env."""
    base = f".lithops/{env}"
    return {
        "euc1_config": f"{base}/config.loaders-euc1.yaml",
        "usw2_config": f"{base}/config.loaders-usw2.yaml",
        "s1_usw2_config": f"{base}/config.s1.yaml",
        "predict_config": f"{base}/config.predict.yaml",
    }


@dataclass
class DownloadConfig:
    runtime: str = "ttc-loaders-dev"
    memory_mb: int = 4096
    retries: int = 3


@dataclass
class PredictConfig:
    runtime: str = "ttc-predict-dev"
    # 6144 MB sized from CloudWatch: observed peak ~3437 MB + ~79% headroom for
    # outlier tiles (larger temporal stacks, denser clouds). Lambda allocates
    # ~3.5 vCPU at this tier vs ~4.6 at 8192 — modest CPU trade for ~25% GB-sec
    # savings. Verify post-change by re-running scripts/predict_lambda_benchmark.py
    # and confirming CloudWatch MaxMemoryUsed stays under ~4500 MB.
    memory_mb: int = 6144
    retries: int = 2
    timeout_sec: int = 600
    model_path: str = "models"


@dataclass
class S1RTCConfig:
    runtime: str = "ttc-s1-dev"
    memory_mb: int = 2048
    retries: int = 3


@dataclass
class ZonalConfig:
    tile_bucket: str = "tof-output"
    tile_region: str = "us-east-1"
    small_sites_area_thresh: float = 0.5
    lulc_raster_path: str = ""
    shift_error_enabled: bool = False
    lookup_parquet: str = "data/tiledb.parquet"
    lookup_csv: str = ""


@dataclass
class LithopsConfig:
    euc1_config: str = ".lithops/config.euc1.yaml"
    usw2_config: str = ".lithops/config.usw2.yaml"
    s1_usw2_config: str = ".lithops/config.s1_usw2.yaml"
    # No default: the old legacy path was us-west-2, which silently cross-regioned
    # writes to `tof-output` (us-east-1). Require explicit opt-in via LITHOPS_ENV or
    # a per-key YAML override so misconfiguration fails loud at the point of use.
    predict_config: str = ""


@dataclass
class PipelineConfig:
    parquet_path: str = "data/tiledb.parquet"
    lithops: LithopsConfig = field(default_factory=LithopsConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    s1_rtc: S1RTCConfig = field(default_factory=S1RTCConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    zonal: ZonalConfig = field(default_factory=ZonalConfig)


def _apply_lithops_env(cfg: PipelineConfig) -> None:
    """If LITHOPS_ENV is set, retarget all four lithops config paths."""
    env = os.environ.get(LITHOPS_ENV_VAR)
    if not env:
        return
    paths = _lithops_paths_for_env(env)
    for key, value in paths.items():
        setattr(cfg.lithops, key, value)
    logger.info(f"{LITHOPS_ENV_VAR}={env} -> using {paths['predict_config']} etc.")


def load_config(path: Optional[str] = None) -> PipelineConfig:
    """Load config from YAML, falling back to defaults for missing keys.

    Precedence for Lithops config paths (lowest to highest):
      1. Built-in defaults (legacy .lithops/config.*.yaml)
      2. LITHOPS_ENV env var -> .lithops/<env>/config.*.yaml
      3. Per-key override in the pipeline YAML
    """
    cfg = PipelineConfig()
    _apply_lithops_env(cfg)

    if path is None:
        default = Path("config.yaml")
        if not default.exists():
            logger.warning("No config file found; using built-in defaults")
            return cfg
        path = str(default)

    logger.info(f"Using config: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    pipeline = raw.get("pipeline", {})
    if pipeline.get("parquet_path"):
        cfg.parquet_path = pipeline["parquet_path"]

    lith = raw.get("lithops", {})
    for key in ("euc1_config", "usw2_config", "s1_usw2_config", "predict_config"):
        if lith.get(key):
            setattr(cfg.lithops, key, lith[key])

    dl = raw.get("download", {})
    for key in ("runtime", "memory_mb", "retries"):
        if dl.get(key) is not None:
            setattr(cfg.download, key, dl[key])

    s1 = raw.get("s1_rtc", {})
    for key in ("runtime", "memory_mb", "retries"):
        if s1.get(key) is not None:
            setattr(cfg.s1_rtc, key, s1[key])

    pred = raw.get("predict", {})
    for key in ("runtime", "memory_mb", "retries", "timeout_sec", "model_path"):
        if pred.get(key) is not None:
            setattr(cfg.predict, key, pred[key])

    zn = raw.get("zonal", {})
    for key in ("tile_bucket", "tile_region", "small_sites_area_thresh",
                "lulc_raster_path", "shift_error_enabled", "lookup_parquet",
                "lookup_csv"):
        if zn.get(key) is not None:
            setattr(cfg.zonal, key, zn[key])

    return cfg
