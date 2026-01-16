"""Telemetry disable flag test."""

import pytest

from src.config.settings import settings
from src.utils import telemetry

pytestmark = pytest.mark.unit


def test_telemetry_disabled_env(tmp_path, monkeypatch):
    out = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(settings.telemetry, "jsonl_path", out)
    monkeypatch.setattr(settings.telemetry, "disabled", True)
    telemetry.log_jsonl({"retrieval.latency_ms": 1})
    assert not out.exists() or out.read_text().strip() == ""
