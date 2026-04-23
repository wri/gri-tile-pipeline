"""Tests for scripts/merge_tof_output_policy.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "merge_tof_output_policy.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("merge_tof_output_policy", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


merge_mod = _load_module()


def _stmt(sid: str, action: str = "s3:GetObject", principal: str = "arn:aws:iam::111111111111:role/foo") -> dict:
    return {
        "Sid": sid,
        "Effect": "Allow",
        "Principal": {"AWS": principal},
        "Action": action,
        "Resource": "arn:aws:s3:::tof-output/*",
    }


def test_append_new_sid():
    current = {"Version": "2012-10-17", "Statement": [_stmt("Existing")]}
    appended = [_stmt("NewOne")]
    merged, summary = merge_mod.merge(current, appended)
    sids = [s["Sid"] for s in merged["Statement"]]
    assert sids == ["Existing", "NewOne"]
    assert summary["added"] == ["NewOne"]
    assert summary["replaced"] == []
    assert summary["untouched"] == ["Existing"]


def test_replace_existing_sid_is_idempotent():
    current = {"Version": "2012-10-17", "Statement": [_stmt("Dup", action="s3:OldAction")]}
    appended = [_stmt("Dup", action="s3:NewAction")]
    merged, summary = merge_mod.merge(current, appended)
    assert len(merged["Statement"]) == 1
    assert merged["Statement"][0]["Action"] == "s3:NewAction"
    assert summary["added"] == []
    assert summary["replaced"] == ["Dup"]


def test_second_run_is_idempotent():
    current = {"Version": "2012-10-17", "Statement": [_stmt("Existing")]}
    appended = [_stmt("CrossAccountRead"), _stmt("CrossAccountWrite")]
    once, _ = merge_mod.merge(current, appended)
    twice, _ = merge_mod.merge(once, appended)
    assert once == twice


def test_cli_writes_output(tmp_path: Path):
    current = tmp_path / "current.json"
    append = tmp_path / "append.json"
    out = tmp_path / "out.json"
    current.write_text(json.dumps({"Version": "2012-10-17", "Statement": [_stmt("Preserved")]}))
    append.write_text(json.dumps([_stmt("NewGrant")]))

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--current", str(current), "--append", str(append), "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    merged = json.loads(out.read_text())
    sids = [s["Sid"] for s in merged["Statement"]]
    assert "Preserved" in sids and "NewGrant" in sids


def test_cli_rejects_missing_sid(tmp_path: Path):
    current = tmp_path / "current.json"
    append = tmp_path / "append.json"
    out = tmp_path / "out.json"
    current.write_text(json.dumps({"Version": "2012-10-17", "Statement": []}))
    append.write_text(json.dumps([{"Effect": "Allow", "Principal": "*", "Action": "s3:GetObject", "Resource": "*"}]))

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--current", str(current), "--append", str(append), "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Sid" in result.stderr


def test_cli_rejects_oversize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    current = tmp_path / "current.json"
    append = tmp_path / "append.json"
    out = tmp_path / "out.json"
    # Build a statement array that will exceed 20 KB after serialization.
    big_value = "x" * 25_000
    current.write_text(json.dumps({"Version": "2012-10-17", "Statement": [_stmt("Big", action=big_value)]}))
    append.write_text(json.dumps([_stmt("Tiny")]))

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--current", str(current), "--append", str(append), "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "20480" in result.stderr or "exceeds" in result.stderr
