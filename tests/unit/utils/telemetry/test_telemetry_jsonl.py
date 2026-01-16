"""Tests for telemetry JSONL writer.

Ensures basic write path works and respects disable/sample envs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.utils.telemetry as telem
from src.config.settings import settings


@pytest.mark.unit
def test_log_jsonl_writes(tmp_path: Path) -> None:
    """Test that log_jsonl writes telemetry data to file when enabled."""
    settings.telemetry.jsonl_path = tmp_path / "telemetry.jsonl"
    settings.telemetry.disabled = False
    settings.telemetry.sample = 1.0
    telem.log_jsonl({"k": 1})
    assert settings.telemetry.jsonl_path.exists()
    assert settings.telemetry.jsonl_path.read_text(encoding="utf-8").strip()


@pytest.mark.unit
def test_log_jsonl_disabled(tmp_path: Path) -> None:
    """Test that log_jsonl respects the telemetry disabled environment variable."""
    settings.telemetry.jsonl_path = tmp_path / "telemetry.jsonl"
    settings.telemetry.disabled = True
    telem.log_jsonl({"k": 1})
    assert not settings.telemetry.jsonl_path.exists()
