"""Reranking configuration propagation tests for the router factory."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval import router_factory as rf


def test_router_propagates_typed_rerank_top_k(
    monkeypatch: pytest.MonkeyPatch, router_settings
) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, tuple[bool, int | None]] = {}

    def _capture(area: str, *, use_reranking: bool, top_n: int | None = None):
        captured[area] = (use_reranking, top_n)
        return []

    class _VectorIndex:
        def as_query_engine(self, **_kwargs):  # type: ignore[no-untyped-def]
            return MagicMock(name="vector_qe")

    class _GraphIndex:
        property_graph_store = object()

    monkeypatch.setattr(rf, "get_postprocessors", _capture, raising=True)
    monkeypatch.setattr(
        "src.retrieval.hybrid.ServerHybridRetriever",
        lambda _params: MagicMock(name="hybrid_retriever"),
        raising=True,
    )
    monkeypatch.setattr(
        rf,
        "build_graph_query_engine",
        lambda *_a, **_k: MagicMock(query_engine=MagicMock(name="graph_qe")),
        raising=True,
    )

    router_settings.retrieval.use_reranking = True
    router_settings.retrieval.reranking_top_k = 7
    router_settings.retrieval.enable_server_hybrid = True

    rf.build_router_engine(
        vector_index=_VectorIndex(),  # type: ignore[arg-type]
        pg_index=_GraphIndex(),  # type: ignore[arg-type]
        settings=router_settings,
    )

    assert captured == {
        "vector": (True, 7),
        "hybrid": (True, 7),
        "kg": (True, 7),
    }
