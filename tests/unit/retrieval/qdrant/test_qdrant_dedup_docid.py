"""Deduplication by doc_id when configured via settings."""

from types import SimpleNamespace

import numpy as np

from src.retrieval.query_engine import ServerHybridRetriever, _HybridParams


def _mk_point(doc_id: str, pid: str, score: float, text: str):
    p = SimpleNamespace()
    p.id = pid
    p.score = score
    p.payload = {"doc_id": doc_id, "page_id": pid, "text": text}
    return p


def test_dedup_by_doc_id(monkeypatch):
    params = _HybridParams(
        collection="c",
        fused_top_k=5,
        prefetch_dense=10,
        prefetch_sparse=10,
        fusion_mode="rrf",
        dedup_key="doc_id",
    )
    retr = ServerHybridRetriever(params)

    monkeypatch.setattr(
        retr,
        "_embed_query",
        lambda s: (np.asarray([0.1, 0.2], dtype=np.float32), {1: 0.5}),
    )

    captured = {}

    class _Res:
        def __init__(self, pts):
            self.points = pts

    def fake_query_points(**kwargs):
        captured.update(kwargs)
        pts = [
            _mk_point("D1", "1", 0.9, "A"),
            _mk_point("D1", "2", 0.8, "A-dup"),
            _mk_point("D2", "3", 0.7, "B"),
        ]
        return _Res(pts)

    monkeypatch.setattr(
        retr._client, "query_points", lambda **kw: fake_query_points(**kw)
    )

    nodes = retr.retrieve("q")
    doc_ids = [n.node.metadata.get("doc_id") for n in nodes]
    assert doc_ids.count("D1") == 1
    # Highest scoring doc D1 remains
    assert any(n.node.text == "A" for n in nodes)
