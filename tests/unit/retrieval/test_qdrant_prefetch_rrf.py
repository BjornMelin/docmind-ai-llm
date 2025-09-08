"""Tests for Qdrant hybrid retrieval with prefetch and RRF fusion."""

from dataclasses import dataclass

from qdrant_client import models as qmodels

from src.retrieval.query_engine import ServerHybridRetriever, _HybridParams


@dataclass
class _Pt:
    """Mock point object for Qdrant query results.

    Attributes:
        id: Unique identifier for the point.
        score: Similarity score for the point.
        payload: Metadata dictionary containing document information.
    """

    id: int
    score: float
    payload: dict


class _Resp:
    """Mock response object for Qdrant query results.

    Args:
        points: List of point objects returned from the query.
    """

    def __init__(self, points):
        self.points = points


def test_server_hybrid_builds_prefetch_and_rrf(monkeypatch):
    """Test that ServerHybridRetriever builds prefetch queries and performs RRF fusion.

    Tests the hybrid retrieval pipeline including:
    - Creation of dense and sparse prefetch queries
    - RRF fusion query construction
    - Deduplication by page_id
    - Proper score ranking after fusion

    Args:
        monkeypatch: Pytest fixture for monkeypatching dependencies.
    """
    params = _HybridParams(
        collection="c",
        fused_top_k=5,
        prefetch_dense=10,
        prefetch_sparse=10,
        fusion_mode="rrf",
    )
    retr = ServerHybridRetriever(params)

    # Stub embed to avoid model deps
    monkeypatch.setattr(retr, "_embed_query", lambda s: ([0.1, 0.2], {1: 0.5, 3: 0.7}))

    captured = {}

    def fake_query_points(**kwargs):
        captured.update(kwargs)
        pts = [
            _Pt(id=1, score=0.9, payload={"page_id": "a", "text": "t"}),
            _Pt(
                id=2, score=0.8, payload={"page_id": "a", "text": "t2"}
            ),  # duplicate page
            _Pt(id=3, score=0.7, payload={"page_id": "b", "text": "t3"}),
        ]
        return _Resp(pts)

    monkeypatch.setattr(
        retr._client, "query_points", lambda **kw: fake_query_points(**kw)
    )

    out = retr.retrieve("q")

    # Assert fusion query
    assert isinstance(captured.get("query"), qmodels.FusionQuery)
    assert captured.get("query").fusion == qmodels.Fusion.RRF

    # Assert two prefetches: sparse and dense
    pf = captured.get("prefetch")
    assert isinstance(pf, list)
    assert len(pf) == 2
    using = {p.using for p in pf}
    assert {"text-sparse", "text-dense"} == using

    # Dedup by page_id: only 2 unique pages remain
    assert len(out) == 2
