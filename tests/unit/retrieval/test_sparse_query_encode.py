"""Tests for sparse query encoding utilities.

Ensures encode_to_qdrant returns a SparseVector when a stub encoder is provided
and returns None when unavailable.
"""

from __future__ import annotations

from qdrant_client import models as qmodels

import src.retrieval.sparse_query as sq


def test_encode_to_qdrant_with_stub(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Patch the sparse encoder to return a stub with indices/values."""

    class _StubEmb:
        def __init__(self) -> None:
            self.indices = [1, 3, 7]
            self.values = [0.2, 0.5, 0.3]

    class _StubEncoder:
        def embed(self, texts):  # type: ignore[no-untyped-def]
            del texts
            return [_StubEmb()]

    monkeypatch.setattr(sq, "_get_sparse_encoder", lambda: _StubEncoder())
    vec = sq.encode_to_qdrant("hello world")
    assert isinstance(vec, qmodels.SparseVector)
    assert vec.indices == [1, 3, 7]
    assert vec.values == [0.2, 0.5, 0.3]


def test_encode_to_qdrant_none(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """If encoder unavailable, function should return None."""
    monkeypatch.setattr(sq, "_get_sparse_encoder", lambda: None)
    assert sq.encode_to_qdrant("x") is None
