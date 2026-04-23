"""Thin worker entry-points for Lithops Lambda invocations.

These must live outside ``gri_tile_pipeline`` so that Lithops can pickle
and unpickle them on Lambda without needing the full package installed
in the Docker image.
"""
from __future__ import annotations

import time
import random
from typing import Any, Dict


def run_dem(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    from loaders.download_dem import run
    return run(**kwargs)


def run_s1_rtc(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """S1 RTC from Planetary Computer (default)."""
    time.sleep(random.uniform(0.0, 2.0))
    from loaders.download_s1_rtc import run
    return run(**kwargs)


def run_s1_legacy(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """S1 GRD from AWS Earth Search (legacy)."""
    from loaders.download_s1 import run
    return run(**kwargs)


def run_s2(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    from loaders.download_s2 import run
    return run(**kwargs)


def run_predict(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    from loaders.predict_tile import run
    return run(**kwargs)
