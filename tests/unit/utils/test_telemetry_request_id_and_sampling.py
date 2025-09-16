"""Unit tests for telemetry request ID and sampling/rotation behavior.

Note: This test redirects the telemetry path to a temporary file and ensures
request_id is included when set.
"""

from __future__ import annotations

import importlib
from pathlib import Path


def test_log_jsonl_includes_request_id(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    tmod = importlib.import_module("src.utils.telemetry")
    # Redirect telemetry path
    monkeypatch.setattr(
        tmod, "_TELEM_PATH", Path(tmp_path / "telemetry.jsonl"), raising=False
    )
    # Ensure sampling logs all
    monkeypatch.setenv("DOCMIND_TELEMETRY_SAMPLE", "1.0")
    # Set request id
    tmod.set_request_id("req-123")
    tmod.log_jsonl({"a": 1})
    data = (
        Path(tmp_path / "telemetry.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert data
    assert "req-123" in data[-1]


def test_log_jsonl_sampling_disables_output(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    tmod = importlib.import_module("src.utils.telemetry")
    monkeypatch.setattr(
        tmod, "_TELEM_PATH", Path(tmp_path / "telemetry.jsonl"), raising=False
    )
    monkeypatch.setenv("DOCMIND_TELEMETRY_SAMPLE", "0.0")
    tmod.set_request_id(None)
    tmod.log_jsonl({"x": 1})
    assert not Path(tmp_path / "telemetry.jsonl").exists()
