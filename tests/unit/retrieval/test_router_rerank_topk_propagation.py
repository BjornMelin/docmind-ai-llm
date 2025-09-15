"""Unit test to ensure reranking_top_k propagates into postprocessor builders.

Monkeypatches get_postprocessors to capture top_n passed from router_factory,
verifying DRY gating and consistent propagation.
"""

from __future__ import annotations

import importlib
from typing import Any


def test_router_rerank_topk_propagation(monkeypatch):  # type: ignore[no-untyped-def]
    rfac = importlib.import_module("src.retrieval.router_factory")

    # Capture calls
    calls: list[tuple[str, bool, int]] = []

    def _fake_get_pp(area: str, *, use_reranking: bool, top_n: int | None = None):  # type: ignore[no-untyped-def]
        calls.append((area, use_reranking, int(top_n or -1)))
        return []

    # Patch retrieval.reranking.get_postprocessors used inside router_factory
    import src.retrieval.reranking as rr

    monkeypatch.setattr(rr, "get_postprocessors", _fake_get_pp, raising=False)

    # Patch ServerHybridRetriever to avoid real client interactions
    import src.retrieval.hybrid as hy

    class _StubRetriever:
        def __init__(self, *_a: Any, **_k: Any):
            pass

        def retrieve(self, *_a: Any, **_k: Any):
            return []

    monkeypatch.setattr(hy, "ServerHybridRetriever", _StubRetriever, raising=False)

    # Configure settings values
    from src.config.settings import settings as cfg

    monkeypatch.setattr(cfg.retrieval, "use_reranking", True, raising=False)
    monkeypatch.setattr(cfg.retrieval, "reranking_top_k", 7, raising=False)
    monkeypatch.setattr(cfg.retrieval, "enable_server_hybrid", True, raising=False)

    # Minimal vector index stub
    class _StubIndex:
        def as_query_engine(self, **_kwargs: Any):  # type: ignore[no-untyped-def]
            class _QE:
                def query(self, *_a: Any, **_k: Any):  # type: ignore[no-untyped-def]
                    return "ok"

            return _QE()

    rfac.build_router_engine(vector_index=_StubIndex(), pg_index=None)
    # We expect calls for at least 'vector' and 'hybrid' tools
    areas = {a for (a, _u, _t) in calls}
    assert "vector" in areas
    assert "hybrid" in areas
    # Validate top_n propagation
    for _a, _u, t in calls:
        if _a in {"vector", "hybrid"}:
            assert t == 7
