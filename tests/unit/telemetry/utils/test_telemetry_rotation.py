"""Tests for telemetry file rotation behavior."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_telemetry_rotation_on_threshold(tmp_path):
    """Rotate telemetry file when exceeding size threshold.

    Creates a file larger than the configured threshold, then writes a log
    event to trigger rotation. Asserts that the rotated file exists.
    """
    from src.config.settings import settings
    from src.utils import telemetry as telemetry_mod

    p = Path(tmp_path) / "t.jsonl"
    p.write_bytes(b"x" * 2048)
    settings.telemetry.jsonl_path = p
    settings.telemetry.disabled = False
    settings.telemetry.sample = 1.0
    settings.telemetry.rotate_bytes = 1024

    telemetry_mod.log_jsonl({"event": "test"})
    assert p.with_suffix(".jsonl.1").exists()
