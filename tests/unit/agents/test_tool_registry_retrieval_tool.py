"""Unit tests for retrieval tool wiring (SPEC-040)."""

from __future__ import annotations

import pytest

from src.agents.registry import DefaultToolRegistry

pytestmark = pytest.mark.unit


def _tool_names(tools) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", "")
        if name:
            names.add(str(name))
    return names


def test_tool_registry_uses_retrieve_documents_for_retrieval_agent() -> None:
    """Ensure retrieval tools include retrieve_documents and exclude router_tool.

    Args:
        None.

    Returns:
        None.
    """
    registry = DefaultToolRegistry()
    names = _tool_names(registry.get_retrieval_tools())
    assert "retrieve_documents" in names
    assert "router_tool" not in names
