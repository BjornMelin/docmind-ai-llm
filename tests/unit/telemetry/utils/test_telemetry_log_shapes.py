"""Telemetry event shape tests via log_jsonl (no file asserts)."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_log_jsonl_sampling_and_disable(tmp_path):
    from src.config.settings import settings
    from src.utils import telemetry as t

    # Redirect path
    p = Path(tmp_path) / "t.jsonl"
    settings.telemetry.jsonl_path = p

    # Disable entirely
    settings.telemetry.disabled = True
    t.log_jsonl({"a": 1})
    assert not p.exists()

    # Enable with sampling 1.0; should write
    settings.telemetry.disabled = False
    settings.telemetry.sample = 1.0
    t.log_jsonl({"b": 2})
    assert p.exists()
