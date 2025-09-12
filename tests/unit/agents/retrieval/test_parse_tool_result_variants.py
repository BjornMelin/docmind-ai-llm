"""Unit tests for tool result parsing and fallback strategy selection."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.unit
def test_parse_tool_result_variants(monkeypatch):
    """Test parsing of various tool result formats into documents."""
    from src.agents import retrieval as mod

    # Minimal agent stub to allow instantiation
    monkeypatch.setattr(
        mod,
        "create_react_agent",
        lambda *_, **__: SimpleNamespace(invoke=lambda *_a, **_k: {"messages": []}),
    )
    agent = mod.RetrievalAgent(llm=None, tools_data={})

    # 1) String result → single minimal document
    docs = agent._parse_tool_result("answer text")
    assert len(docs) == 1
    assert docs[0]["content"] == "answer text"

    # 2) LI-like object with response only
    resp = SimpleNamespace(response="body")
    docs = agent._parse_tool_result(resp)
    assert len(docs) == 1
    assert docs[0]["content"] == "body"

    # 3) LI-like with source_nodes
    node = SimpleNamespace(text="n1", metadata={"a": 1}, score=0.7)
    resp2 = SimpleNamespace(response="ok", source_nodes=[node])
    docs = agent._parse_tool_result(resp2)
    assert len(docs) == 1
    assert docs[0]["metadata"]["a"] == 1

    # 4) List of items with .text/.metadata
    it = SimpleNamespace(text="x", metadata={"k": "v"})
    docs = agent._parse_tool_result([it])
    assert len(docs) == 1
    assert docs[0]["content"] == "x"

    # 5) List of dict-shaped entries
    docs = agent._parse_tool_result(
        [{"content": "c", "metadata": {"m": 1}, "score": 0.1}]
    )
    assert len(docs) == 1
    assert docs[0]["metadata"]["m"] == 1


@pytest.mark.unit
def test_fallback_strategy_selection(monkeypatch):
    """Test fallback strategy selection creates appropriate tools."""
    from src.agents import retrieval as mod

    # Inject a dummy ToolFactory module so internal import picks it up
    tf_mod = ModuleType("src.agents.tool_factory")

    class _StubTool:
        def __init__(self, tag: str):
            self._tag = tag

        def call(self, q: str):  # pylint: disable=unused-argument
            return f"{self._tag}: {q}"

    class ToolFactory:
        @staticmethod
        def create_vector_search_tool(_v):
            return _StubTool("vector")

        @staticmethod
        def create_hybrid_search_tool(_r):
            return _StubTool("hybrid")

        @staticmethod
        def create_kg_search_tool(_kg):
            return _StubTool("kg")

    tf_mod.ToolFactory = ToolFactory  # type: ignore[attr-defined]

    import sys as _sys

    monkeypatch.setitem(_sys.modules, "src.agents.tool_factory", tf_mod)

    # Minimal agent stub
    monkeypatch.setattr(
        mod,
        "create_react_agent",
        lambda *_, **__: SimpleNamespace(invoke=lambda *_a, **_k: {"messages": []}),
    )

    tools_data = {"vector": object(), "retriever": object(), "kg": object()}
    agent = mod.RetrievalAgent(llm=None, tools_data=tools_data)

    # Graphrag path
    out = agent._execute_fallback_retrieval("q", "graphrag")
    assert out["strategy_used"] == "graphrag_fallback"

    # Hybrid path
    out = agent._execute_fallback_retrieval("q", "hybrid")
    assert out["strategy_used"] == "hybrid_fallback"

    # Vector path
    out = agent._execute_fallback_retrieval("q", "vector")
    assert out["strategy_used"] == "vector_fallback"
