"""Unit tests for sparse query encoding in Qdrant format.

Covers encoder missing path and successful encode path with indices/values.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


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


def test_encode_returns_none_when_embedding_has_no_indices_values(monkeypatch):  # type: ignore[no-untyped-def]
    mod = importlib.import_module("src.retrieval.sparse_query")

    class _Emb:
        def __init__(self) -> None:
            self.indices: list[int] = []
            self.values: list[float] = []

    class _Enc:
        def embed(self, _items):  # type: ignore[no-untyped-def]
            yield _Emb()

    monkeypatch.setattr(mod, "_get_sparse_encoder", lambda: _Enc(), raising=False)
    assert mod.encode_to_qdrant("q") is None


def test_get_sparse_encoder_falls_back_to_bm25_when_preferred_fails(monkeypatch):  # type: ignore[no-untyped-def]
    mod = importlib.import_module("src.retrieval.sparse_query")
    mod._get_sparse_encoder.cache_clear()

    calls: list[str] = []

    class SparseTextEmbedding:  # pragma: no cover - small stub
        def __init__(self, model_id: str) -> None:
            calls.append(model_id)
            if model_id == mod.PREFERRED_SPARSE_MODEL:
                raise RuntimeError("offline")

        def embed(self, _items):  # type: ignore[no-untyped-def]
            return iter([])

    fake = ModuleType("fastembed")
    fake.SparseTextEmbedding = SparseTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fake)

    enc = mod._get_sparse_encoder()
    assert enc is not None
    assert calls[:2] == [mod.PREFERRED_SPARSE_MODEL, mod.FALLBACK_SPARSE_MODEL]


def test_get_sparse_encoder_returns_none_when_both_models_fail(monkeypatch):  # type: ignore[no-untyped-def]
    mod = importlib.import_module("src.retrieval.sparse_query")
    mod._get_sparse_encoder.cache_clear()

    class SparseTextEmbedding:  # pragma: no cover - small stub
        def __init__(self, _model_id: str) -> None:
            raise ValueError("bad")

    fake = ModuleType("fastembed")
    fake.SparseTextEmbedding = SparseTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fake)

    assert mod._get_sparse_encoder() is None
