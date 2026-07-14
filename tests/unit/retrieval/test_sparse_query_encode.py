"""Tests for the single pinned BM42 document/query encoding owner."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


class _Embedding:
    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = indices
        self.values = values


def test_document_and_query_callbacks_use_distinct_bm42_algorithms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = importlib.import_module("src.retrieval.sparse_query")
    calls: list[tuple[str, list[str]]] = []

    class _Encoder:
        def embed(self, items: list[str]):  # type: ignore[no-untyped-def]
            calls.append(("documents", items))
            return iter([_Embedding([1], [0.25]) for _ in items])

        def query_embed(self, items: list[str]):  # type: ignore[no-untyped-def]
            calls.append(("queries", items))
            return iter([_Embedding([9], [1.0]) for _ in items])

    monkeypatch.setattr(mod, "_get_sparse_encoder", lambda _cache: _Encoder())
    monkeypatch.setattr(mod, "_cache_folder", lambda: "/models")

    assert mod.encode_documents(["a", "b"]) == ([[1], [1]], [[0.25], [0.25]])
    assert mod.encode_queries(["q"]) == ([[9]], [[1.0]])
    vector = mod.encode_to_qdrant("q")
    assert vector is not None
    assert list(vector.indices) == [9]
    assert list(vector.values) == [1.0]
    assert calls == [
        ("documents", ["a", "b"]),
        ("queries", ["q"]),
        ("queries", ["q"]),
    ]


def test_empty_query_vector_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = importlib.import_module("src.retrieval.sparse_query")

    class _Encoder:
        def query_embed(self, _items: list[str]):  # type: ignore[no-untyped-def]
            return iter([_Embedding([], [])])

    monkeypatch.setattr(mod, "_get_sparse_encoder", lambda _cache: _Encoder())
    assert mod.encode_to_qdrant("the") is None


def test_incomplete_batch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = importlib.import_module("src.retrieval.sparse_query")

    class _Encoder:
        def embed(self, _items: list[str]):  # type: ignore[no-untyped-def]
            return iter([_Embedding([1], [1.0])])

    monkeypatch.setattr(mod, "_get_sparse_encoder", lambda _cache: _Encoder())
    with pytest.raises(mod.SparseEncodingError, match="incomplete batch"):
        mod.encode_documents(["one", "two"])


def test_encoder_resolves_only_the_pinned_local_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = importlib.import_module("src.retrieval.sparse_query")
    mod._get_sparse_encoder.cache_clear()
    snapshot_calls: list[dict[str, object]] = []
    constructor_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(
        mod,
        "snapshot_download",
        lambda **kwargs: snapshot_calls.append(kwargs) or "/models/snapshot",
    )

    class _SparseTextEmbedding:
        def __init__(self, *args: object, **kwargs: object) -> None:
            constructor_calls.append((args, kwargs))

    fake = ModuleType("fastembed")
    fake.SparseTextEmbedding = _SparseTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fake)

    encoder = mod._get_sparse_encoder("/models")

    assert isinstance(encoder, _SparseTextEmbedding)
    assert snapshot_calls == [
        {
            "repo_id": mod.DEFAULT_BM42_SOURCE_REPO,
            "revision": mod.DEFAULT_BM42_SOURCE_REVISION,
            "allow_patterns": list(mod.DEFAULT_BM42_FILES),
            "cache_dir": "/models",
            "local_files_only": True,
        }
    ]
    assert constructor_calls == [
        (
            (mod.DEFAULT_BM42_MODEL_ID,),
            {
                "cache_dir": "/models",
                "specific_model_path": "/models/snapshot",
                "local_files_only": True,
                "providers": ["CPUExecutionProvider"],
            },
        )
    ]
