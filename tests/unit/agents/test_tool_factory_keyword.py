"""Unit tests for optional keyword tool registration behind flag."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.agents.tool_factory import ToolFactory


@pytest.mark.unit
def test_keyword_tool_registration_flag(monkeypatch):
    """Registers keyword tool only when retrieval.enable_keyword_tool=True."""
    # Mock a minimal index
    idx = Mock()
    idx.as_query_engine.return_value = Mock()

    # Ensure disabled by default
    monkeypatch.setattr(
        "src.agents.tool_factory.settings.retrieval.enable_keyword_tool",
        False,
        raising=False,
    )
    tools = ToolFactory.create_tools_from_indexes(idx)
    names = [t.metadata.name for t in tools]
    assert "keyword_search" not in names

    # Enable and verify registration
    monkeypatch.setattr(
        "src.agents.tool_factory.settings.retrieval.enable_keyword_tool",
        True,
        raising=False,
    )
    tools2 = ToolFactory.create_tools_from_indexes(idx)
    names2 = [t.metadata.name for t in tools2]
    assert "keyword_search" in names2
