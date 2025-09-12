import pytest
from qdrant_client import models as qmodels


@pytest.mark.unit
def test_dbsf_fusion_selected(monkeypatch):
    from src.retrieval.hybrid import ServerHybridRetriever, _HybridParams

    class _Res:
        def __init__(self):
            self.points = []

    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        def query_points(self, **kwargs):
            test_dbsf_fusion_selected.kwargs = kwargs
            return _Res()

    monkeypatch.setattr("src.retrieval.hybrid.QdrantClient", _FakeClient)

    retr = ServerHybridRetriever(
        _HybridParams(
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
    kw = test_dbsf_fusion_selected.kwargs
    assert isinstance(kw["query"], qmodels.FusionQuery)
    assert kw["query"].fusion == qmodels.Fusion.DBSF
