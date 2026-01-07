"""Unit tests for router_factory.build_router_engine."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("llama_index.core", reason="requires llama_index.core")
pytest.importorskip(
    "llama_index.program.openai", reason="requires llama_index.program.openai"
)

from src.retrieval.router_factory import build_router_engine

pytestmark = pytest.mark.requires_llama


class _VecIndex:
    def as_query_engine(self):
        return MagicMock(name="vector_qe")


class _HealthyStore:
    def get_nodes(self):
        yield {"id": "n1"}


class _PgIndex:
    def __init__(self, healthy=True) -> None:
        self.property_graph_store = _HealthyStore() if healthy else None

    def as_query_engine(self, include_text=True):
        del include_text
        return MagicMock(name="graph_qe")


def _tool_count(router) -> int:
    # Try public then private attributes; fallback to 0 if not found
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is not None:
            return len(list(tools))
    return 0


@pytest.mark.unit
def test_build_router_engine_with_graph(monkeypatch) -> None:
    vec = _VecIndex()
    pg = _PgIndex(healthy=True)

    def _fake_build_graph_query_engine(*_args, **_kwargs):
        return SimpleNamespace(
            query_engine=MagicMock(name="graph_qe"),
            retriever=MagicMock(name="graph_retriever"),
        )

    monkeypatch.setattr(
        "src.retrieval.router_factory.build_graph_query_engine",
        _fake_build_graph_query_engine,
    )
    monkeypatch.setattr(
        "src.retrieval.reranking.get_postprocessors", lambda *_a, **_k: []
    )
    cfg = SimpleNamespace(
        enable_graphrag=True,
        retrieval=SimpleNamespace(
            top_k=10, use_reranking=False, enable_server_hybrid=False, reranking_top_k=5
        ),
        graphrag_cfg=SimpleNamespace(default_path_depth=1),
        database=SimpleNamespace(qdrant_collection="col"),
    )
    router = build_router_engine(vec, pg, settings=cfg)
    assert _tool_count(router) == 2


@pytest.mark.unit
def test_build_router_engine_vector_only_fallback() -> None:
    vec = _VecIndex()
    pg = _PgIndex(healthy=False)
    cfg = SimpleNamespace(
        enable_graphrag=True,
        retrieval=SimpleNamespace(
            top_k=10, use_reranking=False, enable_server_hybrid=False, reranking_top_k=5
        ),
        graphrag_cfg=SimpleNamespace(default_path_depth=1),
        database=SimpleNamespace(qdrant_collection="col"),
    )
    router = build_router_engine(vec, pg, settings=cfg)
    assert _tool_count(router) == 1
