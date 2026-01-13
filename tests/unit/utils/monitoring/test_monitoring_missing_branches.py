"""Additional unit tests to cover remaining monitoring branches."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.utils import monitoring as mon

pytestmark = pytest.mark.unit


def test_make_otel_log_patcher_handles_non_dict_record() -> None:
    patch = mon._make_otel_log_patcher(enabled=False)
    patch("not-a-dict")


def test_make_otel_log_patcher_normalizes_extra_when_disabled() -> None:
    patch = mon._make_otel_log_patcher(enabled=False)
    record = {"extra": "nope"}
    patch(record)
    assert isinstance(record["extra"], dict)
    assert record["extra"]["otelTraceID"] == ""
    assert record["extra"]["otelSpanID"] == ""
    assert record["extra"]["otelTraceSampled"] == "false"


def test_make_otel_log_patcher_populates_ids_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_mod = ModuleType("opentelemetry.trace")

    class _Ctx:
        is_valid = True
        trace_id = 0x1234
        span_id = 0x5678
        trace_flags = SimpleNamespace(sampled=True)

    class _Span:
        def get_span_context(self):  # type: ignore[no-untyped-def]
            return _Ctx()

    trace_mod.get_current_span = lambda: _Span()  # type: ignore[attr-defined]
    otel = ModuleType("opentelemetry")
    otel.trace = trace_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "opentelemetry", otel)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_mod)

    patch = mon._make_otel_log_patcher(enabled=True)
    record: dict = {"extra": {}}
    patch(record)
    assert record["extra"]["otelTraceID"] == f"{0x1234:032x}"
    assert record["extra"]["otelSpanID"] == f"{0x5678:016x}"
    assert record["extra"]["otelTraceSampled"] == "true"


def test_make_otel_log_patcher_swallows_otel_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_mod = ModuleType("opentelemetry.trace")
    trace_mod.get_current_span = lambda: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[attr-defined]
    otel = ModuleType("opentelemetry")
    otel.trace = trace_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "opentelemetry", otel)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_mod)

    patch = mon._make_otel_log_patcher(enabled=True)
    record: dict = {"extra": {}}
    patch(record)
    # Defaults remain, no exception raised
    assert record["extra"]["otelTraceID"] == ""


def test_setup_logging_includes_trace_prefix_when_observability_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mon,
        "settings",
        SimpleNamespace(
            observability=SimpleNamespace(enabled=True, sampling_ratio=1.0),
            monitoring=SimpleNamespace(bytes_to_mb_divisor=1, percent_multiplier=100),
        ),
        raising=True,
    )

    added: list[dict] = []

    def _add(_sink, **kwargs):  # type: ignore[no-untyped-def]
        added.append(kwargs)

    monkeypatch.setattr(mon.logger, "remove", lambda *_a, **_k: None, raising=True)
    monkeypatch.setattr(mon.logger, "configure", lambda *_a, **_k: None, raising=True)
    monkeypatch.setattr(mon.logger, "add", _add, raising=True)
    monkeypatch.setattr(mon.logger, "info", lambda *_a, **_k: None, raising=True)

    mon.setup_logging(log_level="INFO")
    assert added
    fmt = str(added[0].get("format") or "")
    assert "trace_id={extra[otelTraceID]}" in fmt


def test_log_error_with_context_accepts_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    def fake_error(_msg: str, payload: object):  # type: ignore[no-untyped-def]
        seen.append(payload)

    monkeypatch.setattr(mon.logger, "error", fake_error, raising=True)

    mon.log_error_with_context(ValueError("x"), "op", context={"a": 1}, user="u1")
    assert seen
    payload = seen[0]
    assert isinstance(payload, dict)
    assert payload["operation"] == "op"
    assert payload["user"] == "u1"


def test_log_performance_supports_empty_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_info(_msg: str, payload: dict):  # type: ignore[no-untyped-def]
        calls.append(payload)

    monkeypatch.setattr(mon.logger, "info", fake_info, raising=True)
    mon.log_performance("op", 0.01)
    assert calls
    assert calls[0]["operation"] == "op"


def test_performance_timer_handles_end_memory_probe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mon,
        "settings",
        SimpleNamespace(
            monitoring=SimpleNamespace(bytes_to_mb_divisor=1, percent_multiplier=100)
        ),
        raising=True,
    )

    class _Mem:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class _Proc:
        def __init__(self) -> None:
            self.calls = 0

        def memory_info(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls >= 2:
                raise OSError("boom")
            return _Mem(100)

    monkeypatch.setattr(mon.psutil, "Process", lambda: _Proc(), raising=True)
    monkeypatch.setattr(mon, "log_performance", lambda *_a, **_k: None, raising=True)

    with mon.performance_timer("op") as metrics:
        metrics["x"] = 1
    assert metrics["success"] is True


@pytest.mark.asyncio
async def test_async_performance_timer_handles_psutil_init_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mon,
        "settings",
        SimpleNamespace(
            monitoring=SimpleNamespace(bytes_to_mb_divisor=1, percent_multiplier=100)
        ),
        raising=True,
    )
    monkeypatch.setattr(
        mon.psutil,
        "Process",
        lambda: (_ for _ in ()).throw(OSError("nope")),
        raising=True,
    )
    monkeypatch.setattr(mon, "log_performance", lambda *_a, **_k: None, raising=True)

    async with mon.async_performance_timer("aop") as metrics:
        metrics["k"] = 1
    assert metrics["success"] is True


@pytest.mark.asyncio
async def test_async_performance_timer_handles_end_memory_probe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mon,
        "settings",
        SimpleNamespace(
            monitoring=SimpleNamespace(bytes_to_mb_divisor=1, percent_multiplier=100)
        ),
        raising=True,
    )

    class _Mem:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class _Proc:
        def __init__(self) -> None:
            self.calls = 0

        def memory_info(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls >= 2:
                raise mon.psutil.Error("boom")
            return _Mem(100)

    monkeypatch.setattr(mon.psutil, "Process", lambda: _Proc(), raising=True)
    monkeypatch.setattr(mon, "log_performance", lambda *_a, **_k: None, raising=True)

    async with mon.async_performance_timer("aop2") as metrics:
        metrics["k"] = 2
    assert metrics["success"] is True


def test_simple_monitor_records_summaries_and_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mon,
        "settings",
        SimpleNamespace(
            monitoring=SimpleNamespace(
                bytes_to_mb_divisor=1,
                percent_multiplier=100,
                cpu_monitoring_interval=0.0,
            )
        ),
        raising=True,
    )

    monkeypatch.setattr(mon, "log_performance", lambda *_a, **_k: None, raising=True)
    monkeypatch.setattr(mon.logger, "info", lambda *_a, **_k: None, raising=True)

    m = mon.SimplePerformanceMonitor()
    m.record_operation("op", 1.0, success=True, rows=1)
    m.record_operation("op", 2.0, success=False, rows=2)
    m.record_operation("other", 3.0, success=True, rows=3)

    summary = m.get_summary("op")
    assert summary["total_operations"] == 2
    assert summary["successful_operations"] == 1

    m.clear_metrics()
    assert m.metrics == []
