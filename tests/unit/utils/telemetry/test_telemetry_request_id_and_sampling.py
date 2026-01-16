"""Unit tests for telemetry request ID and sampling/rotation behavior.

Note: This test redirects the telemetry path to a temporary file and ensures
request_id is included when set.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from src.config.settings import settings


def test_log_jsonl_includes_request_id(
    tmp_path: Path,
) -> None:
    """Ensure telemetry log lines include the active request ID."""
    tmod = importlib.import_module("src.utils.telemetry")
    # Redirect telemetry path
    settings.telemetry.jsonl_path = tmp_path / "telemetry.jsonl"
    # Ensure sampling logs all
    settings.telemetry.sample = 1.0
    settings.telemetry.disabled = False
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


def test_log_jsonl_sampling_disables_output(
    tmp_path: Path,
) -> None:
    """Confirm telemetry sampling of 0 suppresses all log output."""
    tmod = importlib.import_module("src.utils.telemetry")
    settings.telemetry.jsonl_path = tmp_path / "telemetry.jsonl"
    settings.telemetry.sample = 0.0
    settings.telemetry.disabled = False
    tmod.set_request_id(None)
    tmod.log_jsonl({"x": 1})
    assert not Path(tmp_path / "telemetry.jsonl").exists()
