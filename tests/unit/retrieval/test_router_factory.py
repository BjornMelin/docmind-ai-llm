"""Unit tests for router_factory.build_router_engine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("llama_index.core", reason="requires llama_index.core")

from src.retrieval.router_factory import build_router_engine

from .conftest import get_router_tool_names

pytestmark = pytest.mark.requires_llama


class _VecIndex:
    def as_query_engine(self, **_kwargs):  # type: ignore[no-untyped-def]
        return MagicMock(name="vector_qe")


class _HealthyStore:
    def get_nodes(self):
        yield {"id": "n1"}


class _PgIndex:
    property_graph_store = _HealthyStore()


@pytest.mark.unit
def test_build_router_engine_with_graph(
    monkeypatch: pytest.MonkeyPatch, router_settings
) -> None:  # type: ignore[no-untyped-def]
    vec = _VecIndex()
    pg = _PgIndex()

    def _fake_build_graph_query_engine(*_args, **_kwargs):
        return MagicMock(query_engine=MagicMock(name="graph_qe"))

    monkeypatch.setattr(
        "src.retrieval.router_factory.build_graph_query_engine",
        _fake_build_graph_query_engine,
    )
    monkeypatch.setattr(
        "src.retrieval.router_factory.get_postprocessors", lambda *_a, **_k: []
    )
    router = build_router_engine(vec, pg, settings=router_settings)
    assert get_router_tool_names(router) == ["semantic_search", "knowledge_graph"]


@pytest.mark.unit
def test_build_router_engine_without_graph(router_settings) -> None:  # type: ignore[no-untyped-def]
    vec = _VecIndex()
    router = build_router_engine(vec, settings=router_settings)
    assert get_router_tool_names(router) == ["semantic_search"]


@pytest.mark.unit
def test_vector_engine_failure_is_not_masked(router_settings) -> None:  # type: ignore[no-untyped-def]
    class _BrokenVectorIndex:
        def as_query_engine(self, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("vector engine failed")

    with pytest.raises(RuntimeError, match="vector engine failed"):
        build_router_engine(_BrokenVectorIndex(), settings=router_settings)
