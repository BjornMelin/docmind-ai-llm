import pytest
from qdrant_client import models as qmodels


@pytest.mark.unit
def test_dbsf_fusion_selected(monkeypatch):
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

    monkeypatch.setattr("src.retrieval.hybrid.QdrantClient", _FakeClient)
    monkeypatch.setattr(
        "src.retrieval.hybrid.ensure_hybrid_collection",
        lambda *_args, **_kwargs: type("_Compatibility", (), {"compatible": True})(),
    )

    retr = ServerHybridRetriever(
        HybridParams(
            collection="test",
            fused_top_k=10,
            prefetch_sparse=2,
            prefetch_dense=2,
            fusion_mode="dbsf",
            dedup_key="page_id",
        )
    )
    monkeypatch.setattr(retr, "_embed_dense", lambda _t: [0.0, 0.1])
    monkeypatch.setattr(
        retr,
        "_encode_sparse",
        lambda _t: qmodels.SparseVector(indices=[3], values=[0.5]),
    )

    _ = retr.retrieve("q")
    kw = calls["kwargs"]
    assert isinstance(kw["query"], qmodels.FusionQuery)
    assert kw["query"].fusion == qmodels.Fusion.DBSF
