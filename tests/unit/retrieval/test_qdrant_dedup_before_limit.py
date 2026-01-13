import pytest


class _P:
    def __init__(self, pid, score, payload):
        self.id = pid
        self.score = score
        self.payload = payload


@pytest.mark.unit
def test_dedup_before_final_truncation(monkeypatch):
    from src.retrieval.hybrid import HybridParams, ServerHybridRetriever

    class _Res:
        def __init__(self, points):
            self.points = points

    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        def query_points(self, **_kwargs):
            pts = [
                _P("a1", 0.9, {"page_id": "p1", "text": "x"}),
                _P("a2", 0.8, {"page_id": "p2", "text": "y"}),
                _P("a3", 0.95, {"page_id": "p1", "text": "x2"}),  # duplicate page_id p1
                _P("a4", 0.7, {"page_id": "p3", "text": "z"}),
            ]
            return _Res(pts)

    monkeypatch.setattr("src.retrieval.hybrid.QdrantClient", _FakeClient)

    retr = ServerHybridRetriever(
        HybridParams(
            collection="test",
            fused_top_k=2,
            prefetch_sparse=5,
            prefetch_dense=5,
            fusion_mode="rrf",
            dedup_key="page_id",
        )
    )
    monkeypatch.setattr(retr, "_embed_dense", lambda _t: [0.0, 0.1])
    monkeypatch.setattr(retr, "_encode_sparse", lambda _t: None)  # dense only

    nodes = retr.retrieve("q")
    # Should dedup p1 keeping higher score (a3) and include next best unique
    assert len(nodes) == 2
    ids = [n.node.id_ for n in nodes]
    # a3 should be present, a1 dropped
    assert "a3" in ids
    assert "a1" not in ids
