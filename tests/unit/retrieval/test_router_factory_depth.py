"""Unit tests for router_factory depth policy and bounded health checks."""

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
    def as_query_engine(self, **_kwargs):  # type: ignore[no-untyped-def]
        return MagicMock(name="vector_qe")


class _PgIndex:
    def __init__(self, depth: int = 1) -> None:
        self.property_graph_store = object()
        self._depth = depth

    def as_query_engine(self, **_kwargs):  # type: ignore[no-untyped-def]
        return MagicMock(name="graph_qe")

    def as_retriever(self, include_text=True, path_depth=1):  # type: ignore[no-untyped-def]
        self.include_text = include_text
        self.path_depth = path_depth
        return MagicMock(name="graph_retriever")


def _tool_count(router) -> int:  # type: ignore[no-untyped-def]
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is not None:
            return len(list(tools))
    return 0


def test_graph_depth_uses_graphrag_cfg(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Graph tool respects settings.graphrag_cfg.default_path_depth."""
    captured: dict[str, object] = {}

    def _fake_build_graph_query_engine(pg_index, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return SimpleNamespace(query_engine=MagicMock(name="graph_qe"))

    monkeypatch.setattr(
        "src.retrieval.router_factory.build_graph_query_engine",
        _fake_build_graph_query_engine,
    )
    monkeypatch.setattr(
        "src.retrieval.reranking.get_postprocessors", lambda *_a, **_k: []
    )

    vec = _VecIndex()
    pg = _PgIndex(depth=3)
    cfg = SimpleNamespace(
        enable_graphrag=True,
        retrieval=SimpleNamespace(
            top_k=8, use_reranking=False, enable_server_hybrid=False, reranking_top_k=4
        ),
        graphrag_cfg=SimpleNamespace(default_path_depth=3),
        database=SimpleNamespace(qdrant_collection="col"),
    )

    build_router_engine(vec, pg, settings=cfg)
    assert captured.get("path_depth") == 3


def test_graph_disabled_when_store_missing() -> None:  # type: ignore[no-untyped-def]
    """No graph tool added when property_graph_store is absent."""
    vec = _VecIndex()

    class _PgMissing:
        property_graph_store = None

    router = build_router_engine(
        vec,
        _PgMissing(),
        settings=SimpleNamespace(
            enable_graphrag=True,
            retrieval=SimpleNamespace(
                top_k=5, use_reranking=False, enable_server_hybrid=False
            ),
            graphrag_cfg=SimpleNamespace(default_path_depth=1),
            database=SimpleNamespace(qdrant_collection="col"),
        ),
    )
    assert _tool_count(router) == 1  # vector only
