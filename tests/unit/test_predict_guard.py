"""Fail-loud guard for the predict step's Lithops config resolution."""

import pytest

from gri_tile_pipeline.steps.predict import _require_predict_config


def test_require_predict_config_raises_on_empty(monkeypatch):
    monkeypatch.delenv("LITHOPS_ENV", raising=False)
    with pytest.raises(RuntimeError, match="LITHOPS_ENV"):
        _require_predict_config("")


def test_require_predict_config_raises_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _require_predict_config(str(missing))


def test_require_predict_config_accepts_existing_file(tmp_path):
    real = tmp_path / "config.predict.yaml"
    real.write_text("lithops: {}\n")
    _require_predict_config(str(real))
