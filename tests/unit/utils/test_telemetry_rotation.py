"""Tests for telemetry file rotation behavior.

Google-Style Docstrings:
    Verifies that when the telemetry file exceeds the configured size, the
    module rotates the file on the next write.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_telemetry_rotation_on_threshold(monkeypatch, tmp_path):
    """Rotate telemetry file when exceeding size threshold.

    Creates a file larger than the configured threshold, then writes a log
    event to trigger rotation. Asserts that the rotated file exists.
    """
    from src.utils import telemetry as telemetry_mod

    p = Path(tmp_path) / "t.jsonl"
    p.write_bytes(b"x" * 2048)
    monkeypatch.setattr(telemetry_mod, "_TELEM_PATH", p)
    monkeypatch.setenv("DOCMIND_TELEMETRY_DISABLED", "false")
    monkeypatch.setenv("DOCMIND_TELEMETRY_SAMPLE", "1.0")
    monkeypatch.setenv("DOCMIND_TELEMETRY_ROTATE_BYTES", "1024")

    telemetry_mod.log_jsonl({"event": "test"})
    assert p.with_suffix(".jsonl.1").exists()
