"""Monitoring log behavior tests using caplog.

Ensures structured context is bound and visible in error/performance logs.
"""

from __future__ import annotations

import pytest

from src.utils import monitoring as mon
from src.utils.monitoring import log_error_with_context, log_performance


@pytest.mark.unit
def test_log_error_with_context_binds_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple, dict]] = []

    def fake_error(
        msg: str, *args: object, **kwargs: object
    ) -> None:  # pragma: no cover - trivial
        calls.append((msg, args, kwargs))

    monkeypatch.setattr(mon.logger, "error", fake_error, raising=True)

    err = ValueError("boom")
    log_error_with_context(err, operation="op1", context={"k": 1})
    assert calls, "logger.error was not called"
    msg, args, _ = calls[0]
    assert "Operation failed" in msg
    assert any("op1" in str(a) for a in args)


@pytest.mark.unit
def test_log_performance_binds_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple, dict]] = []

    def fake_info(
        msg: str, *args: object, **kwargs: object
    ) -> None:  # pragma: no cover - trivial
        calls.append((msg, args, kwargs))

    monkeypatch.setattr(mon.logger, "info", fake_info, raising=True)

    log_performance("op2", 0.123, rows=5)
    assert calls, "logger.info was not called"
    msg, args, _ = calls[0]
    assert "Performance metrics" in msg
    assert any("op2" in str(a) for a in args)
