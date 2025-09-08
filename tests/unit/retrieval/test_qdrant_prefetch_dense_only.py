"""Dense-only prefetch when sparse encoding is unavailable.

Asserts that ServerHybridRetriever builds only a 'text-dense' Prefetch when
the sparse encoder yields no vector, and that no exceptions occur.
"""

from __future__ import annotations

import numpy as np

from src.retrieval.query_engine import ServerHybridRetriever, _HybridParams


class _Resp:
    def __init__(self, points=None):
        self.points = points or []


def test_prefetch_dense_only(monkeypatch):
    params = _HybridParams(
        collection="c",
        fused_top_k=3,
        prefetch_dense=5,
        prefetch_sparse=5,
        fusion_mode="rrf",
    )
    retr = ServerHybridRetriever(params)

    # Stub _embed_query to return dense vector and NO sparse mapping
    monkeypatch.setattr(
        retr, "_embed_query", lambda q: (np.asarray([0.1, 0.2], dtype=np.float32), None)
    )

    captured = {}

    def _fake_query_points(**kw):
        captured.update(kw)
        return _Resp(points=[])

    monkeypatch.setattr(retr._client, "query_points", _fake_query_points)

    retr.retrieve("hello world")

    pf = captured.get("prefetch")
    assert isinstance(pf, list)
    assert len(pf) == 1
    assert pf[0].using == "text-dense"
