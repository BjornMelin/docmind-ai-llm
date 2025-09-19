"""Telemetry disable flag test."""

from src.utils import telemetry


def test_telemetry_disabled_env(tmp_path, monkeypatch):
    out = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(telemetry, "_TELEM_PATH", out, raising=False)
    monkeypatch.setenv("DOCMIND_TELEMETRY_DISABLED", "true")
    telemetry.log_jsonl({"retrieval.latency_ms": 1})
    assert not out.exists() or out.read_text().strip() == ""
