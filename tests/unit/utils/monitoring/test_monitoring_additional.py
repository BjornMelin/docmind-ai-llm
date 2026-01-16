"""Additional coverage for src.utils.monitoring failure paths and logging."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.utils import monitoring as mon


@pytest.mark.unit
def test_performance_timer_failure_path_logs_error() -> None:
    """performance_timer logs metrics with success=False on exceptions."""

    def _run_and_fail():
        with mon.performance_timer("op", extra=1) as m:
            m["k"] = 2
            raise RuntimeError("fail")

    with patch.object(mon, "log_performance") as log_perf:
        with pytest.raises(RuntimeError):
            _run_and_fail()
        assert log_perf.called
    # last call should include success=False and error
    _, kwargs = log_perf.call_args
    assert kwargs.get("success") is False
    assert "error_redacted" in kwargs
    assert "error_fingerprint" in kwargs


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_performance_timer_failure_path_logs_error() -> None:
    """async_performance_timer logs metrics with success=False on exceptions."""

    async def _arun_and_fail():
        async with mon.async_performance_timer("aop", value=3) as m:
            m["x"] = 1
            raise RuntimeError("boom")

    with patch.object(mon, "log_performance") as log_perf:
        with pytest.raises(RuntimeError):
            await _arun_and_fail()
        assert log_perf.called
    _, kwargs = log_perf.call_args
    assert kwargs.get("success") is False
    assert "error_redacted" in kwargs
    assert "error_fingerprint" in kwargs


@pytest.mark.unit
def test_get_memory_usage_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_memory_usage returns zeroed metrics when psutil fails."""

    class _ExplodingProcess:  # pragma: no cover - simple stub
        def memory_info(self):
            raise OSError("denied")

        def memory_percent(self):
            raise OSError("denied")

    monkeypatch.setattr(mon.psutil, "Process", lambda: _ExplodingProcess())

    metrics = mon.get_memory_usage()

    assert metrics == {"rss_mb": 0.0, "vms_mb": 0.0, "percent": 0.0}


@pytest.mark.unit
def test_get_system_info_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_system_info returns empty dict when psutil APIs fail."""

    def _raise(*_args, **_kwargs):  # pragma: no cover - simple stub
        raise mon.psutil.Error("boom")

    monkeypatch.setattr(mon.psutil, "cpu_percent", _raise)
    monkeypatch.setattr(mon.psutil, "virtual_memory", _raise)
    monkeypatch.setattr(mon.psutil, "disk_usage", _raise)
    monkeypatch.setattr(mon.psutil, "getloadavg", _raise, raising=False)

    info = mon.get_system_info()

    assert info == {}


@pytest.mark.unit
def test_performance_timer_handles_psutil_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """performance_timer should not crash when psutil.Process fails."""

    class _ExplodingProcess:  # pragma: no cover - simple stub
        def __init__(self) -> None:
            raise OSError("access denied")

    monkeypatch.setattr(mon.psutil, "Process", _ExplodingProcess)

    with mon.performance_timer("psutil_fail") as metrics:
        metrics["value"] = 1

    # No exception thrown and metrics captured
    assert metrics["value"] == 1
