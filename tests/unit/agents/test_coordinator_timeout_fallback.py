"""Unit tests: timeout handling and fallback in MultiAgentCoordinator.

These tests assert that when the workflow times out (timed_out flag set on
the returned state), the coordinator returns a structured AgentResponse with
fallback_used=True and reason="timeout". When fallback is enabled, a basic
RAG fallback is used; otherwise, a timeout message is returned.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.coordinator import MultiAgentCoordinator
from src.agents.models import AgentResponse


@pytest.mark.unit
def test_timeout_triggers_basic_rag_fallback():
    """Coordinator returns basic RAG fallback on timeout when enabled."""
    coord = MultiAgentCoordinator(enable_fallback=True)

    with (
        patch.object(MultiAgentCoordinator, "_ensure_setup", return_value=True),
        patch.object(
            MultiAgentCoordinator,
            "_run_agent_workflow",
            return_value={"timed_out": True, "messages": []},
        ),
        patch("src.agents.coordinator.AnalyticsManager"),
    ):
        resp = coord.process_query("test query")

    assert isinstance(resp, AgentResponse)
    assert resp.metadata.get("fallback_used") is True
    assert resp.metadata.get("reason") == "timeout"
    assert "unavailable" in resp.content.lower() or "timed out" in resp.content.lower()


@pytest.mark.unit
def test_timeout_without_fallback_returns_timeout_message():
    """Coordinator returns a deterministic timeout message when fallback disabled."""
    coord = MultiAgentCoordinator(enable_fallback=False)

    with (
        patch.object(MultiAgentCoordinator, "_ensure_setup", return_value=True),
        patch.object(
            MultiAgentCoordinator,
            "_run_agent_workflow",
            return_value={"timed_out": True, "messages": []},
        ),
        patch("src.agents.coordinator.AnalyticsManager"),
    ):
        resp = coord.process_query("test query")

    assert isinstance(resp, AgentResponse)
    assert resp.metadata.get("fallback_used") is True
    assert resp.metadata.get("reason") == "timeout"
    assert "timed out" in resp.content.lower()
