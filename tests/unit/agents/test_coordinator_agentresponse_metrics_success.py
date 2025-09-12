"""Unit test: process_query success path includes optimization metrics."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.agents.coordinator import MultiAgentCoordinator
from src.agents.models import AgentResponse


@pytest.mark.unit
def test_process_query_success_returns_optimization_metrics() -> None:
    """Ensure success path yields AgentResponse with optimization metrics keys.

    Mocks the workflow to return a normal (non-timeout) state so that
    _extract_response constructs optimization_metrics from final state.
    """
    coord = MultiAgentCoordinator()

    # Minimal message structure for _extract_response
    msg = SimpleNamespace(content="ok")
    fake_state = {"messages": [msg], "parallel_execution_active": True}

    with (
        patch.object(MultiAgentCoordinator, "_ensure_setup", return_value=True),
        patch.object(
            MultiAgentCoordinator, "_run_agent_workflow", return_value=fake_state
        ),
        patch("src.agents.coordinator.AnalyticsManager"),
    ):
        resp = coord.process_query("hello")

    assert isinstance(resp, AgentResponse)
    om = resp.optimization_metrics
    assert isinstance(om, dict)
    # Check a representative set of keys
    for key in (
        "coordination_overhead_ms",
        "parallel_execution_active",
        "context_trimmed",
        "tokens_trimmed",
        "kv_cache_usage_gb",
        "model_path",
        "optimization_enabled",
    ):
        assert key in om
