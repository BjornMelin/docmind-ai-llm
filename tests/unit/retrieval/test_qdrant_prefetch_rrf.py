import pytest
from qdrant_client import models as qmodels


@pytest.mark.unit
def test_rrf_prefetch_and_limit(monkeypatch):
    from src.retrieval.hybrid import ServerHybridRetriever, _HybridParams

    calls = {}

    class _Res:
        def __init__(self):
            self.points = []

    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        def query_points(self, **kwargs):
            calls["kwargs"] = kwargs
            return _Res()

    # Patch QdrantClient used inside retriever
    monkeypatch.setattr("src.retrieval.hybrid.QdrantClient", _FakeClient)

    retr = ServerHybridRetriever(
        _HybridParams(
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
    assert isinstance(kw["query"], qmodels.FusionQuery)
    # RRF expected
    assert kw["query"].fusion == qmodels.Fusion.RRF
    # Headroom limit = max(prefetch_dense, prefetch_sparse, fused_top_k)
    assert kw["limit"] == 5
