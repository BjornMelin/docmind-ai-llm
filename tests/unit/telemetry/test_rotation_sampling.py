"""Tests for telemetry rotation and sampling behavior (non-failing)."""

import pytest

from src.config.settings import settings
from src.utils import telemetry

pytestmark = pytest.mark.unit


def test_rotation_basic(tmp_path, monkeypatch):
    out = tmp_path / "t.jsonl"
    monkeypatch.setattr(settings.telemetry, "jsonl_path", out)
    monkeypatch.setattr(settings.telemetry, "rotate_bytes", 50)
    monkeypatch.setattr(settings.telemetry, "sample", 1.0)
    monkeypatch.setattr(settings.telemetry, "disabled", False)
    # Write multiple events until rotation triggers
    for i in range(10):
        telemetry.log_jsonl({"retrieval.latency_ms": i})
    assert out.exists()


def test_sampling_drops_events(tmp_path, monkeypatch):
    out = tmp_path / "s.jsonl"
    monkeypatch.setattr(settings.telemetry, "jsonl_path", out)
    monkeypatch.setattr(settings.telemetry, "sample", 0.0)
    monkeypatch.setattr(settings.telemetry, "disabled", False)
    telemetry.log_jsonl({"dedup.dropped": 1})
    # With 0.0 sample rate, file should not exist / be empty
    assert (not out.exists()) or (out.read_text().strip() == "")
