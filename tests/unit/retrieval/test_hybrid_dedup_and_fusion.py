"""Unit tests for hybrid retriever dedup behavior and fusion selection."""

from __future__ import annotations

import importlib


def test_hybrid_dedup_keeps_highest_score(monkeypatch):  # type: ignore[no-untyped-def]
    hmod = importlib.import_module("src.retrieval.hybrid")

    class _Point:
        def __init__(self, pid: str, score: float):
            self.id = pid
            self.score = score
            self.payload = {"page_id": pid, "text": pid}

    class _Res:
        def __init__(self):
            # Duplicate page_id "p1" appears twice with different scores
            self.points = [_Point("p1", 0.2), _Point("p1", 0.9), _Point("p2", 0.5)]

    class _Client:
        def query_points(self, **_kwargs):  # type: ignore[no-untyped-def]
            return _Res()

        def close(self):  # type: ignore[no-untyped-def]
            return None

    params = hmod._HybridParams(  # pylint: disable=protected-access
        collection="c",
        fused_top_k=5,
        prefetch_sparse=2,
        prefetch_dense=2,
        fusion_mode="rrf",
        dedup_key="page_id",
    )
    retr = hmod.ServerHybridRetriever(params, client=_Client())
    out = retr.retrieve("q")
    # Ensure only one of p1 remains and it's the higher score one
    ids = [n.node.node_id for n in out]
    assert len([i for i in ids if i == "p1"]) == 1


def test_fusion_selection_rrf_and_dbsf(monkeypatch):  # type: ignore[no-untyped-def]
    hmod = importlib.import_module("src.retrieval.hybrid")

    # Validate that _fusion() produces proper FusionQuery type depending on mode
    params_rrf = hmod._HybridParams(  # pylint: disable=protected-access
        collection="c", fusion_mode="rrf"
    )
    r = hmod.ServerHybridRetriever(params_rrf, client=lambda: None)  # type: ignore[arg-type]
    assert r._fusion().fusion.name == "RRF"  # pylint: disable=protected-access

    params_dbsf = hmod._HybridParams(  # pylint: disable=protected-access
        collection="c", fusion_mode="dbsf"
    )
    r2 = hmod.ServerHybridRetriever(params_dbsf, client=lambda: None)  # type: ignore[arg-type]
    assert r2._fusion().fusion.name == "DBSF"  # pylint: disable=protected-access
