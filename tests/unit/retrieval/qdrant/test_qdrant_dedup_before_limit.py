"""Assert deduplication by page_id occurs before final cut.

This simulates Qdrant query_points returning multiple points with the same
page_id and verifies that ServerHybridRetriever drops duplicates and keeps
the highest-scoring entry per page_id.
"""

from types import SimpleNamespace

from qdrant_client import models as qmodels

from src.retrieval.query_engine import ServerHybridRetriever, _HybridParams


def _mk_point(pid: str, score: float, payload: dict | None = None):
    p = SimpleNamespace()
    p.id = pid
    p.score = score
    p.payload = payload or {"page_id": pid, "text": f"t-{pid}"}
    return p


def test_dedup_by_page_id_before_final_cut(monkeypatch):
    params = _HybridParams(
        collection="c", fused_top_k=6, prefetch_dense=2, prefetch_sparse=2
    )
    retr = ServerHybridRetriever(params)

    # Stub embeddings (dense + sparse)
    monkeypatch.setattr(retr, "_embed_query", lambda s: ([0.1, 0.2], {1: 0.5}))

    # Simulate Qdrant response with duplicate page_id "A"
    pts = [
        _mk_point("1", 0.9, {"page_id": "A", "text": "A-hi"}),
        _mk_point("2", 0.7, {"page_id": "B", "text": "B-hi"}),
        _mk_point("3", 0.8, {"page_id": "A", "text": "A-lo"}),
        _mk_point("4", 0.6, {"page_id": "C", "text": "C"}),
    ]

    class _Res:
        points = pts

    captured: dict = {}

    def fake_query_points(**kwargs):
        captured.update(kwargs)
        assert isinstance(kwargs.get("query"), qmodels.FusionQuery)
        return _Res()

    monkeypatch.setattr(
        retr._client, "query_points", lambda **kw: fake_query_points(**kw)
    )

    nodes = retr.retrieve("q")
    # Ensure duplicates with same page_id are removed, keeping the higher score
    page_ids = [n.node.metadata.get("page_id") for n in nodes]
    assert page_ids.count("A") == 1
    # The chosen text for A should be the higher-scoring variant "A-hi"
    a_node = next(n for n in nodes if n.node.metadata.get("page_id") == "A")
    assert a_node.node.text == "A-hi"
