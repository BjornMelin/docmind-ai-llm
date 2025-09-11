"""Ensure dedup-before-final-cut enforces uniqueness by configured key.

This complements existing tests by exercising a custom dedup_key.
"""

from types import SimpleNamespace

from qdrant_client import models as qmodels

from src.retrieval.hybrid import ServerHybridRetriever, _HybridParams


def _mk_point(pid: str, did: str, score: float, payload: dict | None = None):
    """Create a mock point object for testing purposes.

    Args:
        pid: Page ID string identifier.
        did: Document ID string identifier.
        score: Relevance score for the point.
        payload: Optional additional payload data to merge with base payload.

    Returns:
        SimpleNamespace object with id, score, and payload attributes.
    """
    p = SimpleNamespace()
    p.id = pid
    p.score = score
    base = {"page_id": pid, "doc_id": did, "text": f"{did}-{pid}"}
    base.update(payload or {})
    p.payload = base
    return p


def test_dedup_unique_by_doc_id(monkeypatch):
    """Test that deduplication works correctly using custom dedup_key.

    This test verifies that when dedup_key is set to "doc_id", points with
    duplicate document IDs are deduplicated, keeping only the highest scoring
    point for each unique document ID.

    Args:
        monkeypatch: pytest fixture for mocking object attributes.
    """
    params = _HybridParams(
        collection="c", fused_top_k=10, fusion_mode="rrf", dedup_key="doc_id"
    )
    retr = ServerHybridRetriever(params)

    # Patch dense/sparse encoders instead of test-only hooks in production code
    monkeypatch.setattr(retr, "_embed_dense", lambda s: [0.1, 0.2])
    monkeypatch.setattr(retr, "_encode_sparse", lambda s: {1: 0.5})

    # Points with duplicate doc_id "D1"; keep highest score only
    pts = [
        _mk_point("1", "D1", 0.9),
        _mk_point("2", "D2", 0.7),
        _mk_point("3", "D1", 0.8),
        _mk_point("4", "D3", 0.6),
    ]

    class _Res:
        points = pts

    captured = {}

    def fake_query_points(**kwargs):
        captured.update(kwargs)
        assert isinstance(kwargs.get("query"), qmodels.FusionQuery)
        return _Res()

    monkeypatch.setattr(
        retr._client, "query_points", lambda **kw: fake_query_points(**kw)
    )

    nodes = retr.retrieve("q")
    doc_ids = [n.node.metadata.get("doc_id") for n in nodes]
    assert doc_ids.count("D1") == 1
