"""Integration test for RouterQueryEngine tools and rerank toggle.

Builds a router via src.retrieval.router_factory with minimal stub indices
and verifies tool registration responds to enable_server_hybrid and
use_reranking flags without raising, and exposes expected tool names.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest


class _StubIndex:
    """Minimal index stub exposing as_query_engine used by router factory."""

    def as_query_engine(self, **_kwargs: Any) -> Any:
        """Return a stub query engine."""

        class _QE:
            """Stub query engine."""

            def query(self, *_a: Any, **_k: Any) -> str:
                """Execute a query and return stub result."""
                return "ok"

        return _QE()


@pytest.mark.integration
@pytest.mark.retrieval
def test_router_tools_and_rerank_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify router tool registration respects hybrid and rerank toggles."""
    rfac = importlib.import_module("src.retrieval.router_factory")

    # Configure settings toggles
    from src.config.settings import settings as cfg

    # Case 1: hybrid off, rerank on
    monkeypatch.setattr(cfg.retrieval, "enable_server_hybrid", False, raising=False)
    monkeypatch.setattr(cfg.retrieval, "use_reranking", True, raising=False)
    router = rfac.build_router_engine(vector_index=_StubIndex(), pg_index=None)
    tools = getattr(router, "query_engine_tools", [])
    names = {t.metadata.name for t in tools}
    assert "semantic_search" in names
    assert "hybrid_search" not in names

    # Case 2: hybrid on
    # Avoid real Qdrant connections by stubbing ServerHybridRetriever
    import src.retrieval.hybrid as hy

    class _StubRetriever:
        """Stub `ServerHybridRetriever` to avoid real Qdrant connections.

        Implements the minimal `retrieve()` surface used by the router factory.
        """

        def __init__(self, *_a: Any, **_k: Any) -> None:
            """Initialize stub retriever."""
            pass

        def close(self) -> None:
            """Close the retriever connection."""
            return None

        def retrieve(self, *_a: Any, **_k: Any) -> list[Any]:
            """Retrieve documents matching the query."""
            return []

    monkeypatch.setattr(hy, "ServerHybridRetriever", _StubRetriever, raising=False)
    monkeypatch.setattr(cfg.retrieval, "enable_server_hybrid", True, raising=False)
    router2 = rfac.build_router_engine(vector_index=_StubIndex(), pg_index=None)
    tools2 = getattr(router2, "query_engine_tools", [])
    names2 = {t.metadata.name for t in tools2}
    assert "semantic_search" in names2
    assert "hybrid_search" in names2
