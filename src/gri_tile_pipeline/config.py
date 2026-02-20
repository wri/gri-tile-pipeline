"""YAML config loading with dataclass defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


@dataclass
class DownloadConfig:
    runtime: str = "ttc-loaders-dev"
    memory_mb: int = 4096
    retries: int = 3


@dataclass
class PredictConfig:
    runtime: str = "ttc-predict-dev"
    memory_mb: int = 8192
    retries: int = 2
    timeout_sec: int = 600


@dataclass
class S1RTCConfig:
    runtime: str = "ttc-s1-dev"
    memory_mb: int = 2048
    retries: int = 3


@dataclass
class ZonalConfig:
    tile_bucket: str = "tof-output"
    small_sites_area_thresh: float = 0.5


@dataclass
class LithopsConfig:
    euc1_config: str = ".lithops/config.euc1.yaml"
    usw2_config: str = ".lithops/config.usw2.yaml"
    s1_usw2_config: str = ".lithops/config.s1_usw2.yaml"
    predict_config: str = ".lithops/config.predict_usw2.yaml"


@dataclass
class PipelineConfig:
    parquet_path: str = "data/tiledb.parquet"
    lithops: LithopsConfig = field(default_factory=LithopsConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    s1_rtc: S1RTCConfig = field(default_factory=S1RTCConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    zonal: ZonalConfig = field(default_factory=ZonalConfig)


def load_config(path: Optional[str] = None) -> PipelineConfig:
    """Load config from YAML, falling back to defaults for missing keys."""
    if path is None:
        # Try default location
        default = Path("config.yaml")
        if not default.exists():
            logger.warning("No config file found; using built-in defaults")
            return PipelineConfig()
        path = str(default)

    logger.info(f"Using config: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = PipelineConfig()

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
    for key in ("runtime", "memory_mb", "retries", "timeout_sec"):
        if pred.get(key) is not None:
            setattr(cfg.predict, key, pred[key])

    zn = raw.get("zonal", {})
    for key in ("tile_bucket", "small_sites_area_thresh"):
        if zn.get(key) is not None:
            setattr(cfg.zonal, key, zn[key])

    return cfg
