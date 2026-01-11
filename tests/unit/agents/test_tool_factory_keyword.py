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

    kw = next(t for t in tools2 if t.metadata.name == "keyword_search")
    desc = (kw.metadata.description or "").lower()
    assert "keyword" in desc
    assert "lexical" in desc or "not semantic" in desc


@pytest.mark.unit
def test_keyword_tool_emits_telemetry_when_sparse_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail open and emit telemetry when sparse encoding is unavailable."""
    # Enable keyword tool
    monkeypatch.setattr(
        "src.agents.tool_factory.settings.retrieval.enable_keyword_tool",
        True,
        raising=False,
    )

    idx = Mock()
    idx.as_query_engine.return_value = Mock()

    events: list[dict] = []
    monkeypatch.setattr("src.retrieval.keyword.log_jsonl", lambda ev: events.append(ev))
    monkeypatch.setattr("src.retrieval.keyword.encode_to_qdrant", lambda _t: None)

    tools = ToolFactory.create_tools_from_indexes(idx)
    kw = next(t for t in tools if t.metadata.name == "keyword_search")

    out = kw.query_engine.query("q")  # exercise BaseQueryEngine interface
    assert out is not None
    assert events
    assert events[-1].get("retrieval.tool") == "keyword_search"
    assert events[-1].get("retrieval.sparse_fallback") is True
