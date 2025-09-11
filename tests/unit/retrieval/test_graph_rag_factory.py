"""Tests for GraphRAG factory that returns store, query_engine, and retriever.

The factory uses only documented LlamaIndex APIs: ``as_query_engine`` and
``as_retriever`` plus the ``property_graph_store`` attribute on the index.
"""

from __future__ import annotations

import pytest

from src.retrieval.graph_config import create_graph_rag_components


class _IndexOK:
    """Stub index exposing documented methods and a graph store attribute."""

    def __init__(self) -> None:
        self.property_graph_store = object()

    def as_query_engine(self, include_text=True, llm=None):
        """Return a stub query engine object."""
        del include_text, llm
        return {"engine": True}

    def as_retriever(self, include_text=False, similarity_top_k=10, path_depth=2):
        """Return a stub retriever object."""
        del include_text, similarity_top_k, path_depth
        return {"retriever": True}


class _IndexNoStore:
    """Stub index missing property_graph_store to trigger validation error."""

    def as_query_engine(self, include_text=True, llm=None):
        """Return a stub query engine object."""
        del include_text, llm
        return {"engine": True}

    def as_retriever(self, include_text=False, similarity_top_k=10, path_depth=2):
        """Return a stub retriever object."""
        del include_text, similarity_top_k, path_depth
        return {"retriever": True}


@pytest.mark.unit
def test_factory_returns_components() -> None:
    """Factory returns graph_store, query_engine, and retriever."""
    idx = _IndexOK()
    out = create_graph_rag_components(
        idx, include_text=True, similarity_top_k=5, path_depth=3
    )
    assert set(out.keys()) == {"graph_store", "query_engine", "retriever"}
    assert out["query_engine"]["engine"] is True
    assert out["retriever"]["retriever"] is True


@pytest.mark.unit
def test_factory_missing_store_raises() -> None:
    """Factory raises when index lacks a property_graph_store attribute."""
    idx = _IndexNoStore()
    with pytest.raises(ValueError, match="property_graph_store"):
        create_graph_rag_components(idx)
