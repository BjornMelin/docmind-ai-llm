"""Additional telemetry tests: disabled, sampling, rotation, and errors."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


def test_disabled_env_no_write(tmp_path, monkeypatch):
    from src.config.settings import settings
    from src.utils import telemetry as t

    tp = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(settings.telemetry, "jsonl_path", tp)
    monkeypatch.setattr(settings.telemetry, "disabled", True)
    t.log_jsonl({"k": 1})
    assert not tp.exists()


def test_sampling_skips_when_rate_zero(tmp_path, monkeypatch):
    from src.config.settings import settings
    from src.utils import telemetry as t

    tp = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(settings.telemetry, "jsonl_path", tp)
    monkeypatch.setattr(settings.telemetry, "disabled", False)
    monkeypatch.setattr(settings.telemetry, "sample", 0.0)
    t.log_jsonl({"k": 2})
    assert not tp.exists()


def test_rotation_and_write(tmp_path, monkeypatch):
    from src.config.settings import settings
    from src.utils import telemetry as t

    tp = tmp_path / "telemetry.jsonl"
    rot = tmp_path / "telemetry.jsonl.1"
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_bytes(b"x" * 10)
    # Set low rotate threshold
    monkeypatch.setattr(settings.telemetry, "jsonl_path", tp)
    monkeypatch.setattr(settings.telemetry, "rotate_bytes", 4)
    monkeypatch.setattr(settings.telemetry, "disabled", False)
    monkeypatch.setattr(settings.telemetry, "sample", 1.0)

    t.log_jsonl({"k": 3})
    # Rotated file exists
    assert rot.exists()
    # New file has one JSON line
    data = tp.read_text(encoding="utf-8").strip().splitlines()
    assert len(data) == 1
    rec = json.loads(data[0])
    assert rec["k"] == 3


def test_rotation_rename_error_is_debug_logged(monkeypatch, tmp_path):
    from src.config.settings import settings
    from src.utils import telemetry as t

    tp = tmp_path / "telemetry.jsonl"
    tp.write_bytes(b"x" * 10)
    monkeypatch.setattr(settings.telemetry, "jsonl_path", tp)
    monkeypatch.setattr(settings.telemetry, "rotate_bytes", 1)
    monkeypatch.setattr(settings.telemetry, "sample", 1.0)
    monkeypatch.setattr(settings.telemetry, "disabled", False)

    def _rename(_self, _dst):
        raise OSError("nope")

    monkeypatch.setattr(type(tp), "rename", _rename, raising=True)

    calls: list[str] = []

    def _dbg(msg: str) -> None:
        calls.append(str(msg))

    monkeypatch.setattr(t.logger, "debug", _dbg, raising=False)
    t.log_jsonl({"k": 4})
    # Should not raise; debug log emitted about rotation skip
    assert any("telemetry rotation skipped" in msg for msg in calls)
