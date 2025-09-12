"""Unit tests: post-model hook attaches metrics regardless of output_mode."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.agents.coordinator import MultiAgentCoordinator


@pytest.mark.unit
def test_post_model_hook_attaches_metrics_without_structured_mode():
    coord = MultiAgentCoordinator()
    hook = coord._create_post_model_hook()
    state = {
        "messages": [Mock(content="hello"), Mock(content="world")],
        "parallel_tool_calls": True,
    }
    out = hook(state)
    assert isinstance(out, dict)
    assert "optimization_metrics" in out
    om = out["optimization_metrics"]
    assert isinstance(om, dict)
    assert om.get("optimization_enabled") is True
    assert "context_used_tokens" in om
    assert "kv_cache_usage_gb" in om
