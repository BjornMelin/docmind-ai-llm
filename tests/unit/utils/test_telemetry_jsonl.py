"""Tests for telemetry JSONL writer.

Ensures basic write path works and respects disable/sample envs.
"""

from __future__ import annotations

import os

import pytest

import src.utils.telemetry as telem


@pytest.mark.unit
def test_log_jsonl_writes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test that log_jsonl writes telemetry data to file when enabled."""
    # Point the telemetry path at a temp file and enable logging
    telem._TELEM_PATH = tmp_path / "telemetry.jsonl"  # type: ignore[attr-defined]
    os.environ.pop("DOCMIND_TELEMETRY_DISABLED", None)
    os.environ["DOCMIND_TELEMETRY_SAMPLE"] = "1.0"
    telem.log_jsonl({"k": 1})
    assert telem._TELEM_PATH.exists()  # type: ignore[attr-defined]
    assert telem._TELEM_PATH.read_text(encoding="utf-8").strip()  # type: ignore[attr-defined]


@pytest.mark.unit
def test_log_jsonl_disabled(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test that log_jsonl respects the telemetry disabled environment variable."""
    telem._TELEM_PATH = tmp_path / "telemetry.jsonl"  # type: ignore[attr-defined]
    os.environ["DOCMIND_TELEMETRY_DISABLED"] = "true"
    telem.log_jsonl({"k": 1})
    assert not telem._TELEM_PATH.exists()  # type: ignore[attr-defined]
    os.environ.pop("DOCMIND_TELEMETRY_DISABLED", None)
