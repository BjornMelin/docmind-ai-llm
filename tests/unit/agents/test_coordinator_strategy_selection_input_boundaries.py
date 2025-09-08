"""Coordinator strategy selection boundary tests with mocked selector."""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


def test_coordinator_handles_unknown_strategy(monkeypatch):
    from src.agents.coordinator import MultiAgentCoordinator

    # Patch internal routing to simulate unknown tool name
    with patch.object(
        MultiAgentCoordinator, "_select_strategy", return_value="unknown"
    ):
        coord = MultiAgentCoordinator(
            model_path="x",
            max_context_length=1024,
            backend="vllm",
            enable_fallback=True,
            max_agent_timeout=1.0,
        )
        # process_query should handle gracefully
        result = coord.process_query("what is AI?", context={})
        assert isinstance(result, dict) or result is not None
