"""Integration test for ingest→router tool composition.

Builds a router with vector and graph tools using stubs to validate composition
and safe fallback behavior.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.retrieval.router_factory import build_router_engine


class _VecIndex:
    def as_query_engine(self):
        """Return a mock vector query engine."""
        return MagicMock(name="vector_qe")


class _HealthyStore:
    def get_nodes(self):
        """Yield a minimal node structure for health checks."""
        yield {"id": "n1"}


class _PgIndex:
    def __init__(self, healthy=True) -> None:
        self.property_graph_store = _HealthyStore() if healthy else None

    def as_query_engine(self, include_text=True):
        """Return a mock graph query engine when store is healthy."""
        del include_text
        return MagicMock(name="graph_qe")


def _tool_count(router) -> int:
    """Return number of tools attached to a RouterQueryEngine instance."""
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is not None:
            return len(list(tools))
    return 0


@pytest.mark.integration
def test_router_composes_vector_and_graph_tools(monkeypatch) -> None:
    """Router includes both vector and graph tools when graph is healthy."""
    vec = _VecIndex()
    pg = _PgIndex(healthy=True)

    monkeypatch.setattr(
        "src.retrieval.router_factory.build_graph_query_engine",
        lambda *_a, **_k: SimpleNamespace(
            query_engine=MagicMock(name="graph_qe"),
            retriever=MagicMock(name="graph_retriever"),
        ),
    )
    monkeypatch.setattr(
        "src.retrieval.reranking.get_postprocessors", lambda *_a, **_k: []
    )

    cfg = SimpleNamespace(
        enable_graphrag=True,
        retrieval=SimpleNamespace(
            top_k=10,
            use_reranking=False,
            enable_server_hybrid=False,
            reranking_top_k=5,
        ),
        graphrag_cfg=SimpleNamespace(default_path_depth=1),
        database=SimpleNamespace(qdrant_collection="col"),
    )
    router = build_router_engine(vec, pg, settings=cfg)
    assert _tool_count(router) == 2


@pytest.mark.integration
def test_router_fallbacks_to_vector_only_when_graph_missing(monkeypatch) -> None:
    """Router falls back to vector-only when graph store is missing."""
    vec = _VecIndex()
    pg = _PgIndex(healthy=False)
    monkeypatch.setattr(
        "src.retrieval.reranking.get_postprocessors", lambda *_a, **_k: []
    )
    router = build_router_engine(vec, pg, settings=None)
    assert _tool_count(router) == 1
