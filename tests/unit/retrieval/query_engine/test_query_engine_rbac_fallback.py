"""RBAC filter fallback tests for query_engine."""

import pytest

pytestmark = pytest.mark.unit


def test_rbac_typed_filter_fallback(monkeypatch):
    """Fallback to dict-based filter when typed FieldCondition raises.

    Patches qmodels.FieldCondition to raise a TypeError; verifies that
    `build_owner_filter` is used instead and query executes with a filter.
    """
    from types import SimpleNamespace

    from src.retrieval import query_engine as qe

    # Dense embedding mock
    class _E:
        def get_query_embedding(self, _t: str):
            return [0.1, 0.2]

    monkeypatch.setattr(qe, "Settings", SimpleNamespace(embed_model=_E()))
    monkeypatch.setenv("DOCMIND_OWNER_ID", "owner-x")

    # Force FieldCondition to raise, triggering fallback
    class _FC:
        def __init__(self, *a, **k):
            raise TypeError("mismatch")

    monkeypatch.setattr(qe.qmodels, "FieldCondition", _FC)

    # Build owner filter returns a dict; patch qmodels.Filter to accept it
    monkeypatch.setattr(
        qe, "build_owner_filter", lambda _owner: {"must": [{"key": "owner_id"}]}
    )
    monkeypatch.setattr(qe.qmodels, "Filter", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(qe.qmodels, "VectorInput", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(
        qe.qmodels,
        "Prefetch",
        lambda query=None, using=None, limit=None: SimpleNamespace(
            query=query, using=using, limit=limit
        ),
    )

    captured = {}

    class _StubClient:
        def query_points(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(points=[])

    monkeypatch.setattr(qe, "QdrantClient", lambda **_k: _StubClient())

    retr = qe.ServerHybridRetriever(qe._HybridParams(collection="col"))
    retr.retrieve("q")
    assert captured.get("query_filter") is not None
