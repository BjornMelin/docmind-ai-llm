"""ServerHybridRetriever prefetch and fusion wiring tests.

Covers:
- Dense and sparse prefetch construction
- Fusion mode selection (RRF vs DBSF)
- RBAC owner filter (typed FieldCondition)
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from qdrant_client import models as qmodels

pytestmark = pytest.mark.unit


def _mk_result(points: list[Any]):
    return SimpleNamespace(points=points)


def _mk_point(pid: str, score: float, payload: dict[str, Any]):
    return SimpleNamespace(id=pid, score=score, payload=payload)


def test_prefetch_and_fusion_mode(monkeypatch):
    # Arrange: patch dense embedder and sparse encoder
    from src.retrieval import query_engine as qe

    # Dense via mocked Settings in module under test
    class _MockEmbed:
        def get_query_embedding(self, _t: str):
            return np.array([0.1, 0.2, 0.3], dtype=np.float32)

    monkeypatch.setattr(qe, "Settings", SimpleNamespace(embed_model=_MockEmbed()))

    # Sparse encoder returns indices/values
    monkeypatch.setattr(
        qe,
        "_encode_sparse_query",
        lambda _t: qmodels.SparseVector(indices=[1, 3], values=[0.5, 0.7]),
    )

    captured = {}

    class _StubClient:
        def query_points(self, **kwargs):
            captured.update(kwargs)
            # Return two dummy points
            pts = [
                _mk_point("a", 0.9, {"text": "one", "page_id": "p1"}),
                _mk_point("b", 0.8, {"text": "two", "page_id": "p2"}),
            ]
            return _mk_result(pts)

    monkeypatch.setattr(qe, "QdrantClient", lambda **_k: _StubClient())
    monkeypatch.setenv("DOCMIND_OWNER_ID", "owner-123")
    # Avoid instantiating typing.Union in older qdrant-client type alias
    monkeypatch.setattr(qe.qmodels, "VectorInput", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(
        qe.qmodels,
        "Prefetch",
        lambda query=None, using=None, limit=None: SimpleNamespace(
            query=query, using=using, limit=limit
        ),
    )

    params = qe._HybridParams(
        collection="col",
        fused_top_k=5,
        prefetch_sparse=111,
        prefetch_dense=222,
        fusion_mode="dbsf",
        dedup_key="page_id",
    )
    retr = qe.ServerHybridRetriever(params)

    # Act
    out = retr.retrieve("hello world")

    # Assert: output shape
    assert isinstance(out, list)
    assert len(out) == 2
    # Prefetch contains both entries using named vectors
    pf = captured["prefetch"]
    assert isinstance(pf, list)
    assert len(pf) == 2
    using = {p.using for p in pf}
    assert using == {"text-sparse", "text-dense"}
    limits = {p.limit for p in pf}
    assert 111 in limits
    assert 222 in limits

    # Fusion mode DBSF when configured
    fusion = captured["query"]
    assert isinstance(fusion, qmodels.FusionQuery)
    assert getattr(fusion, "fusion", None) == qmodels.Fusion.DBSF

    # RBAC filter present and typed
    qf = captured.get("query_filter")
    assert isinstance(qf, qmodels.Filter)
