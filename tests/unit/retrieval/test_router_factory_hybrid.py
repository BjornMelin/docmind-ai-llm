"""Unit tests for router_factory hybrid tool composition."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval import router_factory as rf
from src.retrieval.hybrid import HybridParams

from .conftest import get_router_tool_names


def _pp(mode, *, use_reranking: bool, **kwargs):  # type: ignore[no-untyped-def]
    """Shared postprocessor helper for hybrid tests."""
    del mode, kwargs
    return ["pp"] if use_reranking else None


class _VecIndex:
    def as_query_engine(self, **_kwargs):  # type: ignore[no-untyped-def]
        return MagicMock(name="vector_qe")


def test_hybrid_params_respected(
    monkeypatch: pytest.MonkeyPatch, router_settings
) -> None:  # type: ignore[no-untyped-def]
    """Hybrid retriever receives parameters derived from settings."""
    captured: dict[str, object] = {}

    class _DummyHybrid:
        def __init__(self, params):  # type: ignore[no-untyped-def]
            captured["params"] = params

    monkeypatch.setattr("src.retrieval.hybrid.ServerHybridRetriever", _DummyHybrid)
    monkeypatch.setattr(rf, "get_postprocessors", _pp)

    router_settings.retrieval.enable_server_hybrid = True
    router_settings.retrieval.fused_top_k = 37
    router_settings.retrieval.prefetch_sparse_limit = 222
    router_settings.retrieval.prefetch_dense_limit = 111
    router_settings.retrieval.dedup_key = "doc_id"

    router = rf.build_router_engine(_VecIndex(), settings=router_settings)

    params = captured["params"]
    assert isinstance(params, HybridParams)
    assert params.fused_top_k == 37
    assert params.prefetch_sparse == 222
    assert params.prefetch_dense == 111
    assert params.dedup_key == "doc_id"
    assert get_router_tool_names(router) == ["semantic_search", "hybrid_search"]


@pytest.mark.parametrize("use_rerank", [True, False])
def test_hybrid_rerank_flag_propagation(
    monkeypatch: pytest.MonkeyPatch, router_settings, use_rerank: bool
) -> None:  # type: ignore[no-untyped-def]
    """Hybrid path toggles node_postprocessors when reranking is enabled."""
    last_kwargs: dict[str, object] = {}

    class _DummyHybrid:
        def __init__(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            pass

    class _RecordingRQE:
        @classmethod
        def from_args(cls, **kwargs):  # type: ignore[no-untyped-def]
            last_kwargs.update(kwargs)
            return MagicMock(name="hybrid_qe")

    monkeypatch.setattr("src.retrieval.hybrid.ServerHybridRetriever", _DummyHybrid)
    monkeypatch.setattr(rf, "get_postprocessors", _pp)
    monkeypatch.setattr(rf, "RetrieverQueryEngine", _RecordingRQE)

    router_settings.retrieval.enable_server_hybrid = True
    router_settings.retrieval.use_reranking = use_rerank
    rf.build_router_engine(_VecIndex(), settings=router_settings)

    if use_rerank:
        assert last_kwargs.get("node_postprocessors") is not None
    else:
        assert last_kwargs.get("node_postprocessors") is None
