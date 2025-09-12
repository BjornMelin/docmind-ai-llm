"""Unit tests for RetrievalAgent core behavior.

Covers:
- Success path via stubbed LangGraph agent
- Error path with fallback result
- Performance stats before/after + reset
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _make_stub_agent(payload: dict[str, object]):
    class _Agent:
        def invoke(self, *_a, **_k):  # pylint: disable=unused-argument
            return {
                "messages": [SimpleNamespace(content=json.dumps(payload))],
            }

    return _Agent()


def _raise_agent():
    class _Agent:
        def invoke(self, *_a, **_k):  # pylint: disable=unused-argument
            raise RuntimeError("boom")

    return _Agent()


@pytest.mark.unit
def test_retrieval_agent_success_path(monkeypatch):
    from src.agents import retrieval as mod

    # Stub create_react_agent to a simple agent returning JSON payload
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

    out = agent.retrieve_documents("q", strategy="hybrid", use_dspy=True)
    assert out.document_count == 2
    assert out.strategy_used == "hybrid"
    assert out.query_original == "q"
    assert out.query_optimized == "q*"
    assert out.dspy_used is True
    assert out.graphrag_used is False
    assert isinstance(out.processing_time_ms, float)

    # Strategy accounting updated
    assert agent.strategy_usage.get("hybrid") == 1

    stats = agent.get_performance_stats()
    assert stats["total_retrievals"] == 1
    assert stats["avg_retrieval_time_ms"] >= 0.0


@pytest.mark.unit
def test_retrieval_agent_error_fallback(monkeypatch):
    from src.agents import retrieval as mod

    monkeypatch.setattr(mod, "create_react_agent", lambda *_, **__: _raise_agent())
    agent = mod.RetrievalAgent(llm=None, tools_data={})

    out = agent.retrieve_documents("q", strategy="vector")
    assert out.document_count == 0
    assert out.strategy_used.endswith("_failed")
    assert out.confidence_score == 0.0
    assert "Retrieval failed" in out.reasoning

    # Reset stats clears counters
    agent.reset_stats()
    stats = agent.get_performance_stats()
    assert stats["total_retrievals"] == 0
    assert stats["avg_retrieval_time_ms"] == 0.0
