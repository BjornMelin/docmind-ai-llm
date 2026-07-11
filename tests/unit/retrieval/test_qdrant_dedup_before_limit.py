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
            self.groups = [type("_Group", (), {"hits": [point]})() for point in points]

    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        def query_points_groups(self, **kwargs):
            assert kwargs["group_by"] == "page_id"
            assert kwargs["group_size"] == 1
            assert kwargs["limit"] == 2
            pts = [
                _P("a3", 0.95, {"page_id": "p1", "text": "x2"}),
                _P("a2", 0.8, {"page_id": "p2", "text": "y"}),
            ]
            return _Res(pts)

    monkeypatch.setattr("src.retrieval.hybrid.QdrantClient", _FakeClient)
    monkeypatch.setattr(
        "src.retrieval.hybrid.ensure_hybrid_collection",
        lambda *_args, **_kwargs: type("_Compatibility", (), {"compatible": True})(),
    )

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
    assert len(nodes) == 2
    ids = [n.node.id_ for n in nodes]
    assert "a3" in ids
    assert "a1" not in ids
