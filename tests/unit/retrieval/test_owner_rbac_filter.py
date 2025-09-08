"""RBAC filter: typed FieldCondition when owner env set; dict fallback works."""

import os

from qdrant_client import models as qmodels

from src.retrieval.query_engine import ServerHybridRetriever, _HybridParams


def test_owner_rbac_typed_filter(monkeypatch):
    os.environ["DOCMIND_OWNER_ID"] = "owner-123"
    retr = ServerHybridRetriever(_HybridParams(collection="c"))
    monkeypatch.setattr(retr, "_embed_query", lambda s: ([0.1, 0.2], {1: 0.5}))

    captured = {}

    class _Res:
        def __init__(self) -> None:
            self.points = []

    def fake_query_points(**kwargs):
        captured.update(kwargs)
        return _Res()

    monkeypatch.setattr(
        retr._client, "query_points", lambda **kw: fake_query_points(**kw)
    )
    retr.retrieve("q")

    qf = captured.get("query_filter")
    assert isinstance(qf, qmodels.Filter)
    # must list should be non-empty
    assert getattr(qf, "must", None)


def test_owner_rbac_dict_fallback(monkeypatch):
    os.environ["DOCMIND_OWNER_ID"] = "owner-456"
    retr = ServerHybridRetriever(_HybridParams(collection="c"))
    monkeypatch.setattr(retr, "_embed_query", lambda s: ([0.1, 0.2], {1: 0.5}))

    # Force typed construction path to error by monkeypatching qmodels.FieldCondition
    monkeypatch.setattr(
        qmodels,
        "FieldCondition",
        lambda *a, **k: (_ for _ in ()).throw(TypeError("boom")),
    )

    captured = {}

    class _Res:
        def __init__(self) -> None:
            self.points = []

    def fake_query_points(**kwargs):
        captured.update(kwargs)
        return _Res()

    monkeypatch.setattr(
        retr._client, "query_points", lambda **kw: fake_query_points(**kw)
    )
    retr.retrieve("q")

    qf = captured.get("query_filter")
    # Still a qmodels.Filter, but constructed from dict fallback
    assert isinstance(qf, qmodels.Filter)
