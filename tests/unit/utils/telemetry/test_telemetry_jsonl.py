"""Tests for telemetry JSONL writer.

Ensures basic write path works and respects disable/sample envs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.utils.telemetry as telem
from src.config.settings import settings


@pytest.mark.unit
def test_log_jsonl_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that log_jsonl writes telemetry data to file when enabled."""
    target = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(settings.telemetry, "jsonl_path", target)
    monkeypatch.setattr(settings.telemetry, "disabled", False)
    monkeypatch.setattr(settings.telemetry, "sample", 1.0)
    telem.log_jsonl({"k": 1})
    assert target.exists()
    assert target.read_text(encoding="utf-8").strip()


@pytest.mark.unit
def test_log_jsonl_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that log_jsonl respects the telemetry disabled environment variable."""
    target = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(settings.telemetry, "jsonl_path", target)
    monkeypatch.setattr(settings.telemetry, "disabled", True)
    telem.log_jsonl({"k": 1})
    assert not target.exists()
