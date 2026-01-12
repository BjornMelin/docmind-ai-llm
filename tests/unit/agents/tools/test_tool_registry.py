"""Unit tests for the default tool registry."""

from __future__ import annotations

from src.agents.registry import DefaultToolRegistry


def test_build_tools_data_merges_overrides():
    """The registry should merge overrides into the default payload."""
    registry = DefaultToolRegistry()
    overrides = {"router_engine": "router", "vector": object()}

    tools_data = registry.build_tools_data(overrides)

    assert tools_data["router_engine"] == "router"
    assert "enable_dspy" in tools_data
    assert "vector" not in tools_data


def test_registry_returns_tool_collections():
    """Registry accessors should return non-empty tool collections."""
    registry = DefaultToolRegistry()

    assert registry.get_router_tools(), "router tools expected"
    assert registry.get_planner_tools(), "planner tools expected"
    assert registry.get_retrieval_tools(), "retrieval tools expected"
    assert registry.get_synthesis_tools(), "synthesis tools expected"
    assert registry.get_validation_tools(), "validation tools expected"
