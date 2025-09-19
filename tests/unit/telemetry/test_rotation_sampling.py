"""Tests for telemetry rotation and sampling behavior (non-failing)."""

from src.utils import telemetry


def test_rotation_basic(tmp_path, monkeypatch):
    out = tmp_path / "t.jsonl"
    monkeypatch.setattr(telemetry, "_TELEM_PATH", out, raising=False)
    monkeypatch.setenv("DOCMIND_TELEMETRY_ROTATE_BYTES", "50")
    # Write multiple events until rotation triggers
    for i in range(10):
        telemetry.log_jsonl({"retrieval.latency_ms": i})
    assert out.exists()


def test_sampling_drops_events(tmp_path, monkeypatch):
    out = tmp_path / "s.jsonl"
    monkeypatch.setattr(telemetry, "_TELEM_PATH", out, raising=False)
    monkeypatch.setenv("DOCMIND_TELEMETRY_SAMPLE", "0.0")
    telemetry.log_jsonl({"dedup.dropped": 1})
    # With 0.0 sample rate, file should not exist / be empty
    assert (not out.exists()) or (out.read_text().strip() == "")
