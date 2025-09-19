"""Unit tests for the lean RetrievalAgent wrapper."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


class _DummyTool:
    def __init__(self, payload: dict[str, object]):
        """Expose a langchain-style callable returning the supplied payload."""
        self.func = lambda **_: json.dumps(payload)


def _make_stub_agent(payload: dict[str, object]):
    """Create an agent returning the payload as a JSON message."""

    class _Agent:
        """Stub agent returning the provided payload as JSON."""

        def invoke(self, *_a, **_k):  # pylint: disable=unused-argument
            """Return the stored payload within a LangGraph message envelope."""
            return {"messages": [SimpleNamespace(content=json.dumps(payload))]}

    return _Agent()


def _bad_payload_agent():
    """Return an agent that emits malformed output to trigger fallback."""

    class _Agent:
        """Stub agent returning malformed content for fallback tests."""

        def invoke(self, *_a, **_k):  # pylint: disable=unused-argument
            """Return content that cannot be parsed as JSON."""
            return {"messages": [SimpleNamespace(content="not-json")]}

    return _Agent()


def _raising_agent():
    """Return an agent that raises to exercise error fallback."""

    class _Agent:
        """Stub agent raising to exercise error handling."""

        def invoke(self, *_a, **_k):  # pylint: disable=unused-argument
            """Raise runtime error to mimic agent failure."""
            raise RuntimeError("boom")

    return _Agent()


@pytest.mark.unit
def test_retrieval_agent_success_path(monkeypatch: pytest.MonkeyPatch):
    """Agent invocation returns LangGraph payload and updates metrics."""
    from src.agents import retrieval as mod

    payload = {
        "documents": [
            {"content": "hello", "metadata": {"s": 1}, "score": 0.9},
            {"content": "world", "metadata": {"s": 2}, "score": 0.8},
        ],
        "strategy_used": "hybrid",
        "query_original": "q",
        "query_optimized": "q*",
        "document_count": 2,
        "dspy_used": True,
        "graphrag_used": False,
    }

    monkeypatch.setattr(
        mod, "create_react_agent", lambda *_, **__: _make_stub_agent(payload)
    )

    agent = mod.RetrievalAgent(llm=None, tools_data={})
    assert agent.total_retrievals == 0

    result = agent.retrieve_documents("q", strategy="hybrid", use_dspy=True)
    assert result.document_count == 2
    assert result.strategy_used == "hybrid"
    assert result.query_original == "q"
    assert result.query_optimized == "q*"
    assert result.dspy_used is True
    assert result.graphrag_used is False

    stats = agent.get_performance_stats()
    assert stats["total_retrievals"] == 1
    assert stats["strategy_usage"]["hybrid"] == 1


@pytest.mark.unit
def test_retrieval_agent_falls_back_on_bad_payload(monkeypatch: pytest.MonkeyPatch):
    """Malformed agent output should fall back to the direct tool call."""
    from src.agents import retrieval as mod

    fallback_payload = {
        "documents": [{"content": "fallback", "metadata": {"k": 1}}],
        "strategy_used": "vector",
        "query_original": "q",
        "query_optimized": "q",
    }

    monkeypatch.setattr(
        mod, "create_react_agent", lambda *_, **__: _bad_payload_agent()
    )
    monkeypatch.setattr(mod, "retrieve_documents", _DummyTool(fallback_payload))

    agent = mod.RetrievalAgent(llm=None, tools_data={})
    result = agent.retrieve_documents("q", strategy="vector", use_dspy=False)

    assert result.document_count == 1
    assert result.documents[0]["content"] == "fallback"
    assert result.strategy_used == "vector"
    assert agent.strategy_usage["fallback"] == 1


@pytest.mark.unit
def test_retrieval_agent_error_fallback(monkeypatch: pytest.MonkeyPatch):
    """Agent exceptions propagate through the direct tool fallback."""
    from src.agents import retrieval as mod

    fallback_payload = {
        "documents": [],
        "strategy_used": "vector_failed",
        "query_original": "q",
        "query_optimized": "q",
        "error": "tool crash",
    }

    monkeypatch.setattr(mod, "create_react_agent", lambda *_, **__: _raising_agent())
    monkeypatch.setattr(mod, "retrieve_documents", _DummyTool(fallback_payload))

    agent = mod.RetrievalAgent(llm=None, tools_data={})
    result = agent.retrieve_documents("q", strategy="vector")

    assert result.document_count == 0
    assert result.confidence_score == 0.0
    assert "Retrieval failed" in result.reasoning
    assert agent.strategy_usage["fallback"] == 1

    agent.reset_stats()
    stats = agent.get_performance_stats()
    assert stats["total_retrievals"] == 0
    assert stats["avg_retrieval_time_ms"] == 0.0
