import pytest
from qdrant_client import models as qmodels


@pytest.mark.unit
def test_rrf_prefetch_and_limit(monkeypatch):
    from src.retrieval.hybrid import HybridParams, ServerHybridRetriever

    calls = {}

    class _Res:
        def __init__(self):
            self.groups = []

    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        def query_points_groups(self, **kwargs):
            calls["kwargs"] = kwargs
            return _Res()

    # Patch QdrantClient used inside retriever
    monkeypatch.setattr("src.retrieval.hybrid.QdrantClient", _FakeClient)
    monkeypatch.setattr(
        "src.retrieval.hybrid.check_hybrid_collection",
        lambda *_args, **_kwargs: type("_Compatibility", (), {"compatible": True})(),
    )

    retr = ServerHybridRetriever(
        HybridParams(
            collection="test",
            fused_top_k=3,
            prefetch_sparse=5,
            prefetch_dense=4,
            fusion_mode="rrf",
            dedup_key="page_id",
        )
    )

    # Dense and sparse encoders
    monkeypatch.setattr(retr, "_embed_dense", lambda _t: [0.1, 0.2])
    monkeypatch.setattr(
        retr,
        "_encode_sparse",
        lambda _t: qmodels.SparseVector(indices=[1], values=[1.0]),
    )

    _ = retr.retrieve("q")
    kw = calls["kwargs"]
    assert isinstance(kw["query"], qmodels.RrfQuery)
    assert kw["query"].rrf.k == 60
    assert kw["group_by"] == "page_id"
    assert kw["group_size"] == 1
    assert kw["limit"] == 3
