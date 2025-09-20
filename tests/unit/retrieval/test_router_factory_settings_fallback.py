"""Tests for router_factory hybrid settings fallback behavior.

When ``enable_hybrid`` is ``None`` the router_factory should defer to
``settings.retrieval.enable_server_hybrid``.
"""

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
    """Tiny stub exposing ``as_query_engine`` for vector index."""

    def as_query_engine(self, **_kwargs):  # type: ignore[no-untyped-def]
        return MagicMock(name="vector_qe")


class _HealthyStore:
    def get_nodes(self):  # pragma: no cover - not used in this test
        yield {"id": "n1"}


class _PgIndex:
    """Tiny stub for a property graph index with query engine method."""

    def __init__(self) -> None:
        self.property_graph_store = _HealthyStore()

    def as_query_engine(self, include_text=True):  # type: ignore[no-untyped-def]
        del include_text
        return MagicMock(name="graph_qe")

    def as_retriever(self, include_text=True, path_depth=1):  # type: ignore[no-untyped-def]
        return MagicMock(name="graph_retriever")


def _tool_count(router) -> int:  # type: ignore[no-untyped-def]
    """Return the tool count from a RouterQueryEngine instance.

    Supports both public and private attributes used by LlamaIndex versions.
    """
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is not None:
            return len(list(tools))
    return 0


@pytest.mark.parametrize("flag", [True, False])
def test_build_router_engine_settings_fallback(monkeypatch, flag: bool) -> None:  # type: ignore[no-untyped-def]
    """When ``enable_hybrid`` is ``None``, consult settings flag."""

    class _DummyHybrid:
        def __init__(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            pass

    monkeypatch.setattr("src.retrieval.hybrid.ServerHybridRetriever", _DummyHybrid)

    vec = _VecIndex()
    pg = _PgIndex()
    cfg = SimpleNamespace(
        enable_graphrag=True,
        retrieval=SimpleNamespace(
            top_k=10,
            use_reranking=False,
            enable_server_hybrid=flag,
            reranking_top_k=5,
        ),
        graphrag_cfg=SimpleNamespace(default_path_depth=1),
        database=SimpleNamespace(qdrant_collection="col"),
    )

    router = build_router_engine(vec, pg, settings=cfg, enable_hybrid=None)
    count = _tool_count(router)
    expected = 3 if flag else 2  # vector + graph + optional hybrid
    assert count == expected, f"expected {expected} tools, got {count}"
