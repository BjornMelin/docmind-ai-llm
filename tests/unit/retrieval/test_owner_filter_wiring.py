"""Test that owner RBAC filter is constructed with typed qmodels.Filter."""

from qdrant_client import models as qmodels

from src.retrieval.query_engine import ServerHybridRetriever, _HybridParams


def test_owner_filter_constructed(monkeypatch):
    params = _HybridParams(
        collection="c", fused_top_k=3, prefetch_dense=1, prefetch_sparse=1
    )
    retr = ServerHybridRetriever(params)

    # Stub embeddings
    monkeypatch.setenv("DOCMIND_OWNER_ID", "owner-42")
    import numpy as np
    monkeypatch.setattr(
        retr, "_embed_query", lambda s: (np.asarray([0.1, 0.2], dtype=np.float32), {})
    )

    captured = {}

    class _Res:
        def __init__(self) -> None:
            self.points: list = []

    def fake_query_points(**kwargs):
        captured.update(kwargs)
        return _Res()

    monkeypatch.setattr(
        retr._client, "query_points", lambda **kw: fake_query_points(**kw)
    )

    retr.retrieve("q")
    qf = captured.get("query_filter")
    assert isinstance(qf, qmodels.Filter)
    assert qf.must
    has_owner = any(
        isinstance(c, qmodels.FieldCondition) and c.key == "owner_id" for c in qf.must
    )
    assert has_owner
