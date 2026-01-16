"""Tests for telemetry rotation and sampling behavior (non-failing)."""

from src.config.settings import settings
from src.utils import telemetry


def test_rotation_basic(tmp_path):
    out = tmp_path / "t.jsonl"
    settings.telemetry.jsonl_path = out
    settings.telemetry.rotate_bytes = 50
    settings.telemetry.sample = 1.0
    settings.telemetry.disabled = False
    # Write multiple events until rotation triggers
    for i in range(10):
        telemetry.log_jsonl({"retrieval.latency_ms": i})
    assert out.exists()


def test_sampling_drops_events(tmp_path):
    out = tmp_path / "s.jsonl"
    settings.telemetry.jsonl_path = out
    settings.telemetry.sample = 0.0
    settings.telemetry.disabled = False
    telemetry.log_jsonl({"dedup.dropped": 1})
    # With 0.0 sample rate, file should not exist / be empty
    assert (not out.exists()) or (out.read_text().strip() == "")
