"""Tests for sparse query encoding via FastEmbed wrapper."""

import pytest

pytestmark = pytest.mark.unit


def test_encode_to_qdrant_fastembed_available(monkeypatch):
    """Return SparseVector when FastEmbed encoder yields indices/values."""
    from qdrant_client import models as qmodels

    from src.retrieval import sparse_query as sq

    class _Emb:
        def __init__(self):
            self.indices = [1, 5]
            self.values = [0.5, 0.7]

    class _Enc:
        def query_embed(self, texts):
            return [_Emb() for _ in texts]

    monkeypatch.setattr(sq, "_get_sparse_encoder", lambda _cache: _Enc())
    out = sq.encode_to_qdrant("hello")
    assert isinstance(out, qmodels.SparseVector)
    assert out.indices == [1, 5]
    assert out.values == [0.5, 0.7]


def test_encode_to_qdrant_encoder_failure_propagates(monkeypatch):
    """Normalize provider failures for query-time fallback handling."""
    from src.retrieval import sparse_query as sq

    def _unavailable(_cache):  # type: ignore[no-untyped-def]
        raise RuntimeError("encoder unavailable")

    monkeypatch.setattr(sq, "_get_sparse_encoder", _unavailable)
    with pytest.raises(sq.SparseEncodingError, match="query encoding failed") as exc:
        sq.encode_to_qdrant("x")
    assert isinstance(exc.value.__cause__, RuntimeError)
