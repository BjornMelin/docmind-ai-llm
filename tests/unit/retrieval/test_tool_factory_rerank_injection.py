"""Tests for reranker injection in ToolFactory.

Verifies that node_postprocessors are attached when
settings.retrieval.use_reranking is True, and omitted when False.
"""

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("llama_index.core", reason="requires llama_index.core")

pytestmark = [pytest.mark.unit, pytest.mark.requires_llama]


class _FakeIndex:
    """Fake index for testing."""

    def __init__(self):
        self.calls = []

    def as_query_engine(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(qe=True, kwargs=kwargs)


class _FakeRetriever:
    """Fake retriever for testing."""

    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):  # pragma: no cover - not used
        self.kwargs = kwargs


def test_vector_hybrid_tools_inject_reranker_when_enabled(monkeypatch):
    """Test that reranker is injected when enabled."""
    tf = importlib.import_module("src.agents.tool_factory")
    import src.retrieval.reranking as rr

    # Ensure setting is True
    monkeypatch.setattr(tf.settings.retrieval, "use_reranking", True, raising=False)
    monkeypatch.setattr(
        rr, "get_postprocessors", lambda *_a, **_k: [SimpleNamespace()], raising=True
    )

    idx = _FakeIndex()

    # Vector search
    tool_v = tf.ToolFactory.create_vector_search_tool(idx)
    assert tool_v.query_engine.kwargs.get("node_postprocessors"), (
        "vector missing reranker"
    )

    # Hybrid vector search fallback
    tool_h = tf.ToolFactory.create_hybrid_vector_tool(idx)
    assert tool_h.query_engine.kwargs.get("node_postprocessors"), (
        "hybrid missing reranker"
    )


def test_kg_tool_injects_text_reranker_when_enabled(monkeypatch):
    """Test that reranker is injected when enabled."""
    tf = importlib.import_module("src.agents.tool_factory")
    import src.retrieval.reranking as rr

    monkeypatch.setattr(tf.settings.retrieval, "use_reranking", True, raising=False)
    monkeypatch.setattr(
        rr, "get_postprocessors", lambda *_a, **_k: [SimpleNamespace()], raising=True
    )

    class _KG:
        def __init__(self):
            self.kwargs = None

        def as_query_engine(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(qe=True, kwargs=kwargs)

    kg = _KG()
    tool = tf.ToolFactory.create_kg_search_tool(kg)
    assert tool.query_engine.kwargs.get("node_postprocessors"), "kg missing reranker"


def test_tools_do_not_inject_when_disabled(monkeypatch):
    """Test that reranker is not injected when disabled."""
    tf = importlib.import_module("src.agents.tool_factory")
    import src.retrieval.reranking as rr

    monkeypatch.setattr(tf.settings.retrieval, "use_reranking", False, raising=False)
    monkeypatch.setattr(rr, "get_postprocessors", lambda *_a, **_k: None, raising=True)
    idx = _FakeIndex()
    tool_v = tf.ToolFactory.create_vector_search_tool(idx)
    assert tool_v.query_engine.kwargs.get("node_postprocessors") is None
    tool_h = tf.ToolFactory.create_hybrid_vector_tool(idx)
    assert tool_h.query_engine.kwargs.get("node_postprocessors") is None
