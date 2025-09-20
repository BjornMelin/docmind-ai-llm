"""Unit tests for router_factory hybrid tool composition."""

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


class _HealthyStore:
    def get_nodes(self):  # pragma: no cover - unused branch helper
        yield {"id": "n1"}


class _PgIndex:
    def __init__(self) -> None:
        self.property_graph_store = _HealthyStore()

    def as_query_engine(self, include_text=True):  # type: ignore[no-untyped-def]
        del include_text
        return MagicMock(name="graph_qe")


def test_hybrid_params_respected(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Hybrid retriever receives parameters derived from settings."""
    captured: dict[str, SimpleNamespace] = {}

    class _DummyHybrid:
        def __init__(self, params):  # type: ignore[no-untyped-def]
            captured["params"] = params

    monkeypatch.setattr("src.retrieval.hybrid.ServerHybridRetriever", _DummyHybrid)

    vec = _VecIndex()
    pg = _PgIndex()
    cfg = SimpleNamespace(
        enable_graphrag=True,
        retrieval=SimpleNamespace(
            top_k=5,
            use_reranking=True,
            enable_server_hybrid=False,
            fused_top_k=37,
            prefetch_sparse_limit=222,
            prefetch_dense_limit=111,
            fusion_mode="rrf",
            dedup_key="doc_id",
            reranking_top_k=3,
        ),
        graphrag_cfg=SimpleNamespace(default_path_depth=1),
        database=SimpleNamespace(qdrant_collection="col"),
    )

    build_router_engine(vec, pg, settings=cfg, enable_hybrid=True)

    params = captured["params"]
    assert params.fused_top_k == 37
    assert params.prefetch_sparse == 222
    assert params.prefetch_dense == 111
    assert params.dedup_key == "doc_id"


@pytest.mark.parametrize("use_rerank", [True, False])
def test_hybrid_rerank_flag_propagation(monkeypatch, use_rerank: bool) -> None:  # type: ignore[no-untyped-def]
    """Hybrid path toggles node_postprocessors when reranking is enabled."""
    last_kwargs: dict[str, object] = {}

    class _DummyHybrid:
        def __init__(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            pass

    class _RecordingRQE:
        @classmethod
        def from_args(cls, **kwargs):  # type: ignore[no-untyped-def]
            last_kwargs.update(kwargs)
            return SimpleNamespace(qe=True, kwargs=kwargs)

    monkeypatch.setattr("src.retrieval.hybrid.ServerHybridRetriever", _DummyHybrid)
    monkeypatch.setattr(
        "src.retrieval.router_factory.build_retriever_query_engine",
        lambda retriever, post, **kwargs: _RecordingRQE.from_args(
            retriever=retriever, node_postprocessors=post, **kwargs
        ),
    )

    vec = _VecIndex()
    pg = _PgIndex()
    cfg = SimpleNamespace(
        enable_graphrag=True,
        retrieval=SimpleNamespace(
            top_k=5,
            use_reranking=use_rerank,
            enable_server_hybrid=False,
            reranking_top_k=2,
        ),
        graphrag_cfg=SimpleNamespace(default_path_depth=1),
        database=SimpleNamespace(qdrant_collection="col"),
    )

    build_router_engine(vec, pg, settings=cfg, enable_hybrid=True)

    if use_rerank:
        assert last_kwargs.get("node_postprocessors") is not None
    else:
        assert last_kwargs.get("node_postprocessors") is None
