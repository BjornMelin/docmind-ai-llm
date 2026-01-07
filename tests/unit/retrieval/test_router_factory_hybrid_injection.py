"""RouterFactory hybrid rerank injection tests (enable_hybrid=True).

Verifies that when reranking is enabled, the hybrid tool receives
node_postprocessors; and when disabled, no node_postprocessors are passed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.retrieval import router_factory as rf

from .conftest import get_router_tool_names


@pytest.mark.unit
def test_hybrid_rerank_injection_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _DummyHybrid:
        def __init__(self, *_a: object, **_k: object) -> None:
            return None

    def _fake_build_retriever_query_engine(retriever, post, **kwargs):  # type: ignore[no-untyped-def]
        del retriever
        captured["post"] = post
        return SimpleNamespace(qe=True, kwargs=kwargs, post=post)

    monkeypatch.setattr(
        "src.retrieval.hybrid.ServerHybridRetriever", _DummyHybrid, raising=False
    )
    monkeypatch.setattr(
        "src.retrieval.router_factory.build_retriever_query_engine",
        _fake_build_retriever_query_engine,
        raising=True,
    )
    monkeypatch.setattr(
        "src.retrieval.router_factory.build_graph_query_engine",
        lambda *_a, **_k: SimpleNamespace(query_engine=SimpleNamespace()),
        raising=True,
    )

    def _fake_get_postprocessors(_kind: str, *, use_reranking: bool, **_kwargs: object):
        return ["pp"] if use_reranking else None

    monkeypatch.setattr(
        "src.retrieval.reranking.get_postprocessors",
        _fake_get_postprocessors,
        raising=False,
    )

    class _Vec:
        def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(qe=True, kwargs=kwargs)

    class _Pg:
        def __init__(self) -> None:
            self.property_graph_store = object()

    cfg = SimpleNamespace(
        enable_graphrag=True,
        retrieval=SimpleNamespace(
            top_k=5,
            use_reranking=True,
            reranking_top_k=3,
            enable_server_hybrid=False,
            fused_top_k=10,
            prefetch_sparse_limit=100,
            prefetch_dense_limit=50,
            fusion_mode="rrf",
            dedup_key="page_id",
        ),
        graphrag_cfg=SimpleNamespace(default_path_depth=1),
        database=SimpleNamespace(qdrant_collection="c"),
    )

    router = rf.build_router_engine(
        _Vec(), pg_index=_Pg(), settings=cfg, enable_hybrid=True
    )
    assert set(get_router_tool_names(router)) >= {
        "semantic_search",
        "hybrid_search",
        "knowledge_graph",
    }
    assert captured.get("post") == ["pp"]

    cfg.retrieval.use_reranking = False
    captured.clear()
    router2 = rf.build_router_engine(
        _Vec(), pg_index=_Pg(), settings=cfg, enable_hybrid=True
    )
    assert set(get_router_tool_names(router2)) >= {
        "semantic_search",
        "hybrid_search",
        "knowledge_graph",
    }
    assert captured.get("post") is None
