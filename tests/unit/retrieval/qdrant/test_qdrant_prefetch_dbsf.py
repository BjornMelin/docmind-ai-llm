"""Tests for Qdrant hybrid retrieval with DBSF fusion mode."""

from qdrant_client import models as qmodels

from src.retrieval.query_engine import ServerHybridRetriever, _HybridParams


def test_server_hybrid_uses_dbsf_when_configured(monkeypatch):
    """Test that ServerHybridRetriever uses DBSF fusion when configured.

    Verifies that when fusion_mode is set to "dbsf", the hybrid retriever
    correctly constructs a FusionQuery with DBSF fusion method.

    Args:
        monkeypatch: Pytest fixture for monkeypatching dependencies.
    """
    params = _HybridParams(
        collection="c",
        fused_top_k=5,
        prefetch_dense=10,
        prefetch_sparse=10,
        fusion_mode="dbsf",
    )
    retr = ServerHybridRetriever(params)

    # Stub embedder
    monkeypatch.setattr(retr, "_embed_query", lambda s: ([0.1, 0.2], {1: 0.5}))

    captured = {}

    def fake_query_points(**kwargs):
        captured.update(kwargs)
        return type("R", (), {"points": []})

    monkeypatch.setattr(
        retr._client, "query_points", lambda **kw: fake_query_points(**kw)
    )

    retr.retrieve("q")
    assert isinstance(captured.get("query"), qmodels.FusionQuery)
    assert captured.get("query").fusion == qmodels.Fusion.DBSF
