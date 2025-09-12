"""Coordinator timeout metrics and analytics behavior tests.

Validates that:
- successful_queries is not incremented on timeouts
- fallback_queries increments only when fallback is used
- analytics logs exactly once with success=False on timeout
- analytics logs exactly once with success=True on success path
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from src.agents.coordinator import MultiAgentCoordinator
from src.agents.models import AgentResponse


class _FakeAM:
    """Fake AnalyticsManager that records log_query calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def log_query(self, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
        """Record analytics log payload for assertions."""
        self.calls.append(dict(kwargs))


@pytest.mark.unit
def test_timeout_counters_and_analytics_single_log_with_fallback(
    tmp_path, monkeypatch
):  # type: ignore[no-untyped-def]
    """Timeout: do not count as success; log one failure entry."""
    coord = MultiAgentCoordinator(enable_fallback=True)

    fake_am = _FakeAM()

    # Enable analytics and stub manager instance
    import src.agents.coordinator as mod

    monkeypatch.setattr(
        mod,
        "settings",
        SimpleNamespace(  # toggle analytics
            analytics_enabled=True,
            analytics_db_path=None,
            data_dir=tmp_path,
            analytics_retention_days=7,
            vllm=SimpleNamespace(context_window=131072, max_tokens=1024),
            retrieval=SimpleNamespace(
                reranker_normalize_scores=False, reranking_top_k=5, top_k=5
            ),
        ),
        raising=False,
    )

    with (
        patch.object(MultiAgentCoordinator, "_ensure_setup", return_value=True),
        patch.object(
            MultiAgentCoordinator,
            "_run_agent_workflow",
            return_value={"timed_out": True, "messages": []},
        ),
        patch.object(
            MultiAgentCoordinator,
            "_fallback_basic_rag",
            return_value=AgentResponse(
                content="fallback",
                sources=[],
                metadata={"fallback_used": True, "reason": "timeout"},
                validation_score=0.0,
                processing_time=0.01,
                optimization_metrics={"timeout": True},
            ),
        ),
        patch(
            "src.agents.coordinator.AnalyticsManager.instance",
            return_value=fake_am,
        ),
    ):
        resp = coord.process_query("q")

    assert isinstance(resp, AgentResponse)
    # Counters: total increments, fallback increments, success does not
    assert coord.total_queries == 1
    assert coord.fallback_queries == 1
    assert coord.successful_queries == 0
    # Analytics: exactly one log with success=False
    assert len(fake_am.calls) == 1
    assert fake_am.calls[0].get("success") is False


@pytest.mark.unit
def test_success_counters_and_analytics_single_log(
    monkeypatch,
):  # type: ignore[no-untyped-def]
    """Success path: count as success; log one success entry."""
    coord = MultiAgentCoordinator(enable_fallback=True)

    fake_am = _FakeAM()

    import src.agents.coordinator as mod

    monkeypatch.setattr(
        mod,
        "settings",
        SimpleNamespace(
            analytics_enabled=True,
            analytics_db_path=None,
            data_dir=mod.settings.data_dir,  # reuse configured path
            analytics_retention_days=7,
            vllm=SimpleNamespace(context_window=131072, max_tokens=1024),
            retrieval=SimpleNamespace(
                reranker_normalize_scores=False, reranking_top_k=5, top_k=5
            ),
        ),
        raising=False,
    )

    final_state = {
        "messages": [SimpleNamespace(content="ok")],
        "synthesis_result": {"documents": []},
    }

    with (
        patch.object(MultiAgentCoordinator, "_ensure_setup", return_value=True),
        patch.object(
            MultiAgentCoordinator, "_run_agent_workflow", return_value=final_state
        ),
        patch("src.agents.coordinator.AnalyticsManager.instance", return_value=fake_am),
    ):
        resp = coord.process_query("q")

    assert isinstance(resp, AgentResponse)
    # Counters: total increments, success increments, fallback stays 0
    assert coord.total_queries == 1
    assert coord.successful_queries == 1
    assert coord.fallback_queries == 0
    # Analytics: exactly one log with success=True
    assert len(fake_am.calls) == 1
    assert fake_am.calls[0].get("success") is True


@pytest.mark.unit
def test_timeout_no_fallback_counters_and_failure_analytics(
    tmp_path, monkeypatch
):  # type: ignore[no-untyped-def]
    """Timeout without fallback: no success, no fallback increment, 1 failure log."""
    coord = MultiAgentCoordinator(enable_fallback=False)

    fake_am = _FakeAM()

    import src.agents.coordinator as mod

    monkeypatch.setattr(
        mod,
        "settings",
        SimpleNamespace(
            analytics_enabled=True,
            analytics_db_path=None,
            data_dir=tmp_path,
            analytics_retention_days=7,
            vllm=SimpleNamespace(context_window=131072, max_tokens=1024),
            retrieval=SimpleNamespace(
                reranker_normalize_scores=False, reranking_top_k=5, top_k=5
            ),
        ),
        raising=False,
    )

    with (
        patch.object(MultiAgentCoordinator, "_ensure_setup", return_value=True),
        patch.object(
            MultiAgentCoordinator,
            "_run_agent_workflow",
            return_value={"timed_out": True, "messages": []},
        ),
        patch(
            "src.agents.coordinator.AnalyticsManager.instance",
            return_value=fake_am,
        ),
    ):
        resp = coord.process_query("q")

    assert isinstance(resp, AgentResponse)
    assert resp.metadata.get("reason") == "timeout"
    # Counters: total increments; success stays 0; fallback stays 0
    assert coord.total_queries == 1
    assert coord.successful_queries == 0
    assert coord.fallback_queries == 0
    # Analytics: exactly one log with success=False
    assert len(fake_am.calls) == 1
    assert fake_am.calls[0].get("success") is False
