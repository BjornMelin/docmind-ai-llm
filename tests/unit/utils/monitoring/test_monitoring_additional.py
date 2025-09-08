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
    assert "error" in kwargs


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
    assert "error" in kwargs
