"""Unit test to ensure reranking_top_k propagates into postprocessor builders.

Monkeypatches get_postprocessors to capture top_n passed from router_factory,
verifying DRY gating and consistent propagation.
"""

from __future__ import annotations

import importlib
from typing import Any


def _run_router_with_capture(
    monkeypatch, *, rerank_top_k: Any, enable_hybrid: bool = True
):
    rfac = importlib.import_module("src.retrieval.router_factory")
    captured: dict[str, tuple[bool, int | None]] = {}

    def _fake_get_pp(area: str, *, use_reranking: bool, top_n: int | None = None):
        normalized = int(top_n) if top_n is not None else None
        captured[area] = (use_reranking, normalized)
        return []

    import src.retrieval.reranking as rr

    monkeypatch.setattr(rr, "get_postprocessors", _fake_get_pp, raising=False)

    import src.retrieval.hybrid as hy

    class _StubRetriever:
        def __init__(self, *_a: Any, **_k: Any):
            pass

        def retrieve(self, *_a: Any, **_k: Any):
            return []

    monkeypatch.setattr(hy, "ServerHybridRetriever", _StubRetriever, raising=False)

    from src.config.settings import settings as cfg

    monkeypatch.setattr(cfg.retrieval, "use_reranking", True, raising=False)
    monkeypatch.setattr(cfg.retrieval, "reranking_top_k", rerank_top_k, raising=False)
    monkeypatch.setattr(
        cfg.retrieval, "enable_server_hybrid", enable_hybrid, raising=False
    )

    class _StubIndex:
        def as_query_engine(self, **_kwargs: Any):
            class _QE:
                def query(self, *_a: Any, **_k: Any):
                    return "ok"

            return _QE()

    class _StubGraph:
        def as_retriever(self, **_kwargs: Any):
            class _Retriever:
                def retrieve(self, *_args: Any, **_kwargs: Any):
                    return []

            return _Retriever()

    rfac.build_router_engine(vector_index=_StubIndex(), pg_index=_StubGraph())
    return captured


def test_router_rerank_topk_propagation(monkeypatch):  # type: ignore[no-untyped-def]
    captured = _run_router_with_capture(monkeypatch, rerank_top_k=7)
    assert captured.get("vector") == (True, 7)
    assert captured.get("hybrid") == (True, 7)
    assert captured.get("kg") == (True, 7)


def test_router_rerank_topk_none(monkeypatch):  # type: ignore[no-untyped-def]
    captured = _run_router_with_capture(monkeypatch, rerank_top_k=None)
    assert captured.get("vector") == (True, None)
    assert captured.get("hybrid") == (True, None)
    assert captured.get("kg") == (True, None)


def test_router_rerank_topk_invalid(monkeypatch):  # type: ignore[no-untyped-def]
    captured = _run_router_with_capture(monkeypatch, rerank_top_k="15")
    assert captured.get("vector") == (True, 15)
    assert captured.get("hybrid") == (True, 15)
    assert captured.get("kg") == (True, 15)
