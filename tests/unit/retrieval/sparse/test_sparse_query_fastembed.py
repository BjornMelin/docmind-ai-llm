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
        def embed(self, texts):
            return [_Emb() for _ in texts]

    monkeypatch.setattr(sq, "_get_sparse_encoder", lambda: _Enc())
    out = sq.encode_to_qdrant("hello")
    assert isinstance(out, qmodels.SparseVector)
    assert out.indices == [1, 5]
    assert out.values == [0.5, 0.7]


def test_encode_to_qdrant_encoder_unavailable(monkeypatch):
    """Return None when encoder is unavailable (ImportError)."""
    from src.retrieval import sparse_query as sq

    monkeypatch.setattr(sq, "_get_sparse_encoder", lambda: None)
    assert sq.encode_to_qdrant("x") is None
