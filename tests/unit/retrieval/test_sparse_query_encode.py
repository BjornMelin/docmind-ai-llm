"""Unit tests for sparse query encoding in Qdrant format.

Covers encoder missing path and successful encode path with indices/values.
"""

from __future__ import annotations

import importlib


def test_encode_returns_none_when_encoder_missing(monkeypatch):  # type: ignore[no-untyped-def]
    mod = importlib.import_module("src.retrieval.sparse_query")

    # Force cache to bypass real import and return None encoder
    monkeypatch.setattr(mod, "_get_sparse_encoder", lambda: None, raising=False)
    out = mod.encode_to_qdrant("query")
    assert out is None


def test_encode_returns_sparse_vector(monkeypatch):  # type: ignore[no-untyped-def]
    mod = importlib.import_module("src.retrieval.sparse_query")

    class _Emb:
        def __init__(self):
            self.indices = [1, 5, 9]
            self.values = [0.5, 0.3, 0.2]

    class _Enc:
        def embed(self, _items):  # type: ignore[no-untyped-def]
            yield _Emb()

    monkeypatch.setattr(mod, "_get_sparse_encoder", lambda: _Enc(), raising=False)
    out = mod.encode_to_qdrant("hello")
    assert out is not None
    assert list(out.indices) == [1, 5, 9]
    assert list(out.values) == [0.5, 0.3, 0.2]
