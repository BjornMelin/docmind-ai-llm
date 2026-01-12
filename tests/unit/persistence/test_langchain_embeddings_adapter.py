"""Unit tests for LlamaIndexEmbeddingsAdapter (LangChain Embeddings bridge)."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from src.persistence.langchain_embeddings import LlamaIndexEmbeddingsAdapter

pytestmark = pytest.mark.unit


def test_embed_query_and_documents_with_stubbed_llamaindex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_mod = ModuleType("llama_index.core")

    class _FakeEmbedModel:
        def get_text_embedding(self, text: str):  # type: ignore[no-untyped-def]
            return [1.0, 2.0, float(len(text))]

        def get_text_embedding_batch(self, texts):  # type: ignore[no-untyped-def]
            return [[1.0, 0.0, float(len(t))] for t in texts]

    class _Settings:
        embed_model = _FakeEmbedModel()

    core_mod.Settings = _Settings
    monkeypatch.setitem(sys.modules, "llama_index.core", core_mod)

    emb = LlamaIndexEmbeddingsAdapter()
    q = emb.embed_query("hi")
    assert q == [1.0, 2.0, 2.0]

    docs = emb.embed_documents(["a", "bb"])
    assert docs == [[1.0, 0.0, 1.0], [1.0, 0.0, 2.0]]
