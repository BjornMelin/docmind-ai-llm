"""Tests for the lightweight JSONL telemetry emitter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils import telemetry


@pytest.mark.unit
def test_log_jsonl_respects_disable_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(telemetry, "_TELEM_PATH", target)
    monkeypatch.setenv("DOCMIND_TELEMETRY_DISABLED", "true")
    telemetry.log_jsonl({"event": "disabled"})
    assert not target.exists()


@pytest.mark.unit
def test_log_jsonl_writes_event(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(telemetry, "_TELEM_PATH", target)
    monkeypatch.delenv("DOCMIND_TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("DOCMIND_TELEMETRY_SAMPLE", "1.0")
    telemetry.set_request_id("req-42")

    telemetry.log_jsonl({"event": "test", "value": 3})

    data = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
    telemetry.set_request_id(None)
    assert data["event"] == "test"
    assert data["request_id"] == "req-42"


@pytest.mark.unit
def test_log_jsonl_rotates_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "telemetry.jsonl"
    rotated = target.with_suffix(".jsonl.1")
    target.write_text("old\n", encoding="utf-8")

    monkeypatch.setattr(telemetry, "_TELEM_PATH", target)
    monkeypatch.setenv("DOCMIND_TELEMETRY_SAMPLE", "1.0")
    monkeypatch.setenv("DOCMIND_TELEMETRY_ROTATE_BYTES", "4")

    telemetry.log_jsonl({"event": "rotate"})

    assert rotated.exists()
    assert target.exists()
    assert "rotate" in target.read_text(encoding="utf-8")
