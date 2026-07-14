"""Unit tests for canonical worker tool-set construction."""

from __future__ import annotations

import pytest

from src.agents.registry import build_agent_tool_sets
from src.config.settings import DocMindSettings


def test_builder_returns_four_worker_tool_sets() -> None:
    """The production builder returns one non-empty set per worker."""
    groups = build_agent_tool_sets()

    assert set(groups) == {
        "planner_agent",
        "retrieval_agent",
        "synthesis_agent",
        "validation_agent",
    }
    assert all(groups.values())


def _tool_names(tools) -> set[str]:
    names = set()
    for tool in tools:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", "")
        if name:
            names.add(str(name))
    return names


@pytest.mark.unit
def test_registry_includes_ollama_web_tools_when_enabled() -> None:
    # Build cfg with nested security override to avoid in-place mutation
    cfg = DocMindSettings(
        ollama_enable_web_search=True,
        ollama_api_key="key-123",
        security={"allow_remote_endpoints": True},
    )
    names = _tool_names(build_agent_tool_sets(cfg)["retrieval_agent"])
    assert "ollama_web_search" in names
    assert "ollama_web_fetch" in names


@pytest.mark.unit
def test_registry_excludes_ollama_web_tools_when_disabled() -> None:
    cfg = DocMindSettings(ollama_enable_web_search=False)
    names = _tool_names(build_agent_tool_sets(cfg)["retrieval_agent"])
    assert "ollama_web_search" not in names
    assert "ollama_web_fetch" not in names
