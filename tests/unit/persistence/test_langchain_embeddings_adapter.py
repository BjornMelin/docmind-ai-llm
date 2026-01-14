from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from src.persistence.langchain_embeddings import LlamaIndexEmbeddingsAdapter

pytestmark = pytest.mark.unit


def _set_embed_model(monkeypatch, model: object | None) -> None:
    import importlib

    li_core = importlib.import_module("llama_index.core")
    # Patch the module-level Settings reference so we can control embed_model
    # without triggering BaseEmbedding validation.
    monkeypatch.setattr(
        li_core,
        "Settings",
        SimpleNamespace(embed_model=model),
        raising=False,
    )


def test_embed_query_requires_embed_model(monkeypatch) -> None:
    _set_embed_model(monkeypatch, None)
    emb = LlamaIndexEmbeddingsAdapter()
    with pytest.raises(RuntimeError, match="embed_model is not configured"):
        emb.embed_query("hi")


def test_embed_query_delegates_to_llamaindex(monkeypatch) -> None:
    class _Model:
        def get_text_embedding(self, text: str) -> Sequence[float]:
            return [1, 2.5, 3.0]

    _set_embed_model(monkeypatch, _Model())
    emb = LlamaIndexEmbeddingsAdapter()
    assert emb.embed_query("hi") == [1.0, 2.5, 3.0]


def test_embed_documents_empty_returns_empty(monkeypatch) -> None:
    _set_embed_model(monkeypatch, SimpleNamespace(get_text_embedding=lambda _t: [1.0]))
    emb = LlamaIndexEmbeddingsAdapter()
    assert emb.embed_documents([]) == []


def test_embed_documents_prefers_batch_method(monkeypatch) -> None:
    class _Model:
        def get_text_embedding_batch(self, texts: list[str]) -> list[Sequence[float]]:
            assert texts == ["a", "b"]
            return [[1, 2], [3, 4]]

    _set_embed_model(monkeypatch, _Model())
    emb = LlamaIndexEmbeddingsAdapter()
    assert emb.embed_documents(["a", "b"]) == [[1.0, 2.0], [3.0, 4.0]]


def test_embed_documents_falls_back_to_text_embeddings(monkeypatch) -> None:
    class _Model:
        def get_text_embeddings(self, texts: list[str]) -> list[Sequence[float]]:
            assert texts == ["a", "b"]
            return [[1, 2], [3, 4]]

    _set_embed_model(monkeypatch, _Model())
    emb = LlamaIndexEmbeddingsAdapter()
    assert emb.embed_documents(["a", "b"]) == [[1.0, 2.0], [3.0, 4.0]]


def test_embed_documents_falls_back_to_single_embedding(monkeypatch) -> None:
    class _Model:
        def get_text_embedding(self, text: str) -> Sequence[float]:
            return [float(len(text))]

    _set_embed_model(monkeypatch, _Model())
    emb = LlamaIndexEmbeddingsAdapter()
    assert emb.embed_documents(["a", "bb"]) == [[1.0], [2.0]]
