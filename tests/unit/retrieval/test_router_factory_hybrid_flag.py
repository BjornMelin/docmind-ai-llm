"""Coverage tests for router_factory hybrid tool flag behavior.

Ensure that when enable_hybrid=True is passed, a hybrid tool is registered,
increasing the tool count by one.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.retrieval.router_factory import build_router_engine


class _VecIndex:
    def as_query_engine(self, **_kwargs):  # type: ignore[no-untyped-def]
        return MagicMock(name="vector_qe")


class _HealthyStore:
    def get_nodes(self):  # pragma: no cover - not used
        yield {"id": "n1"}


class _PgIndex:
    def __init__(self) -> None:
        self.property_graph_store = _HealthyStore()

    def as_query_engine(self, include_text=True):  # type: ignore[no-untyped-def]
        del include_text
        return MagicMock(name="graph_qe")


def _tool_count(router) -> int:  # type: ignore[no-untyped-def]
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is not None:
            return len(list(tools))
    return 0


def test_build_router_engine_with_hybrid_flag(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """When enable_hybrid=True, the router includes a hybrid_search tool."""
    # Patch hybrid retriever where it is defined (avoid exposing test-only symbols)

    class _DummyHybrid:
        def __init__(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            pass

    monkeypatch.setattr("src.retrieval.hybrid.ServerHybridRetriever", _DummyHybrid)

    vec = _VecIndex()
    pg = _PgIndex()

    monkeypatch.setattr(
        "src.retrieval.router_factory.build_graph_query_engine",
        lambda *_a, **_k: SimpleNamespace(
            query_engine=MagicMock(name="graph_qe"),
            retriever=MagicMock(name="graph_retriever"),
        ),
    )

    cfg = SimpleNamespace(
        enable_graphrag=True,
        retrieval=SimpleNamespace(
            top_k=5,
            use_reranking=True,
            enable_server_hybrid=False,
            reranking_top_k=3,
        ),
        graphrag_cfg=SimpleNamespace(default_path_depth=1),
        database=SimpleNamespace(qdrant_collection="c"),
    )

    router = build_router_engine(vec, pg, settings=cfg, enable_hybrid=True)
    # Expect vector + kg + hybrid
    assert _tool_count(router) == 3
