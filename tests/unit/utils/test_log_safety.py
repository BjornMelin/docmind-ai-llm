"""Unit tests for safe logging helpers (SPEC-028 / NFR-SEC-002)."""

from __future__ import annotations

from io import StringIO

import pytest
from loguru import logger

from src.utils.log_safety import (
    fingerprint_text,
    redact_text_backstop,
    safe_url_for_log,
)
from src.utils.monitoring import log_error_with_context

pytestmark = pytest.mark.unit


def test_fingerprint_text_is_deterministic() -> None:
    a = fingerprint_text("hello@example.com", key_id="k1")
    b = fingerprint_text("hello@example.com", key_id="k1")
    assert a == b
    assert a["len"] == len("hello@example.com")
    assert isinstance(a["hmac_sha256_12"], str)
    assert len(str(a["hmac_sha256_12"])) == 12


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com/path?x=1", "https://example.com"),
        ("http://localhost:8000/v1/models", "http://localhost:8000"),
        ("https://user:pass@example.com/v1/models", "https://example.com"),
        ("http://[::1]:8000/v1/models", "http://[::1]:8000"),
        ("http://example.com:bad/v1/models", "http://example.com"),
        ("not-a-url", ""),
        ("", ""),
    ],
)
def test_safe_url_for_log(url: str, expected: str) -> None:
    assert safe_url_for_log(url) == expected


def test_redact_text_backstop_redacts_common_secret_patterns() -> None:
    raw = "Authorization: Bearer SECRET1234567890 sk-abcdefghijklmno"
    out = redact_text_backstop(raw)
    assert "SECRET1234567890" not in out
    assert "sk-abcdefghijklmno" not in out
    assert "[redacted" in out


def test_canary_string_does_not_appear_in_logs_or_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = "CANARY_SECRET_12345"
    captured_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        "src.utils.monitoring.log_jsonl",
        lambda event: captured_events.append(dict(event)),
    )

    sink = StringIO()
    sink_id = logger.add(sink, format="{message}", level="DEBUG")
    try:
        log_error_with_context(
            RuntimeError(canary),
            operation="test_canary",
            context={"user_input": canary},
        )
    finally:
        logger.remove(sink_id)

    assert canary not in sink.getvalue()
    assert captured_events
    for event in captured_events:
        for value in event.values():
            if isinstance(value, str):
                assert canary not in value
