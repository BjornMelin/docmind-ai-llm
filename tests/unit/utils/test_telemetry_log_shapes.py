"""Telemetry event shape tests via log_jsonl (no file asserts)."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_log_jsonl_sampling_and_disable(monkeypatch, tmp_path):
    from src.utils import telemetry as t

    # Redirect path
    monkeypatch.setattr(t, "_TELEM_PATH", Path(tmp_path) / "t.jsonl")

    # Disable entirely
    monkeypatch.setenv("DOCMIND_TELEMETRY_DISABLED", "true")
    t.log_jsonl({"a": 1})
    assert not t._TELEM_PATH.exists()

    # Enable with sampling 1.0; should write
    monkeypatch.setenv("DOCMIND_TELEMETRY_DISABLED", "false")
    monkeypatch.setenv("DOCMIND_TELEMETRY_SAMPLE", "1.0")
    t.log_jsonl({"b": 2})
    assert t._TELEM_PATH.exists()
