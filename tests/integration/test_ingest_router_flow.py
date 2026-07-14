"""Integration test for ingest→router tool composition.

Builds a router with vector and graph tools using stubs to validate composition.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from llama_index.core.query_engine import RouterQueryEngine

from src.retrieval.router_factory import build_router_engine

pytest.importorskip("llama_index.core", reason="requires llama_index.core")

pytestmark = pytest.mark.requires_llama


class _VecIndex:
    def as_query_engine(self, **_kwargs):  # type: ignore[no-untyped-def]
        """Return a mock vector query engine."""
        return MagicMock(name="vector_qe")


class _HealthyStore:
    def get_nodes(self):
        """Yield a minimal node structure for health checks."""
        yield {"id": "n1"}


class _PgIndex:
    def __init__(self, healthy=True) -> None:
        self.property_graph_store = _HealthyStore() if healthy else None


def _tool_names(router: RouterQueryEngine) -> set[str]:
    """Return the registered router tool names."""
    return {metadata.name for metadata in router._metadatas}


@pytest.mark.integration
def test_router_composes_vector_and_graph_tools(
    monkeypatch: pytest.MonkeyPatch, integration_settings
) -> None:  # type: ignore[no-untyped-def]
    """Router includes both vector and graph tools when graph is healthy."""
    vec = _VecIndex()
    pg = _PgIndex(healthy=True)

    monkeypatch.setattr(
        "src.retrieval.router_factory.build_graph_query_engine",
        lambda *_a, **_k: MagicMock(query_engine=MagicMock(name="graph_qe")),
    )
    monkeypatch.setattr(
        "src.retrieval.router_factory.get_postprocessors", lambda *_a, **_k: []
    )

    integration_settings.retrieval.enable_image_retrieval = False
    integration_settings.retrieval.enable_server_hybrid = False
    integration_settings.retrieval.enable_keyword_tool = False
    integration_settings.retrieval.use_reranking = False
    router = build_router_engine(vec, pg, settings=integration_settings)
    assert _tool_names(router) == {"semantic_search", "knowledge_graph"}


@pytest.mark.integration
def test_router_uses_vector_only_when_graph_missing(
    monkeypatch: pytest.MonkeyPatch, integration_settings
) -> None:  # type: ignore[no-untyped-def]
    """Router uses vector-only composition when no graph index exists."""
    vec = _VecIndex()
    monkeypatch.setattr(
        "src.retrieval.router_factory.get_postprocessors", lambda *_a, **_k: []
    )
    integration_settings.retrieval.enable_image_retrieval = False
    integration_settings.retrieval.enable_server_hybrid = False
    integration_settings.retrieval.enable_keyword_tool = False
    router = build_router_engine(vec, settings=integration_settings)
    assert _tool_names(router) == {"semantic_search"}
