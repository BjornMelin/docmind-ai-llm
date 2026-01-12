"""LangChain Embeddings adapters for DocMind (SPEC-041).

LangGraph stores expect LangChain `Embeddings`. DocMind already configures
LlamaIndex `Settings.embed_model` from unified settings; this adapter bridges
that embed model into LangChain.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from langchain_core.embeddings import Embeddings


class LlamaIndexEmbeddingsAdapter(Embeddings):
    """LangChain Embeddings wrapper around LlamaIndex Settings.embed_model."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string into a float vector."""
        from llama_index.core import Settings as LISettings

        model = LISettings.embed_model
        vec = cast(Sequence[float], model.get_text_embedding(str(text)))
        return [float(x) for x in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents into float vectors."""
        from llama_index.core import Settings as LISettings

        model = LISettings.embed_model
        batch = [str(t) for t in texts]
        get_batch = getattr(model, "get_text_embedding_batch", None)
        if callable(get_batch):
            vecs = cast(list[Sequence[float]], get_batch(batch))
            return [[float(x) for x in v] for v in vecs]

        get_text_embeddings = getattr(model, "get_text_embeddings", None)
        if callable(get_text_embeddings):
            vecs = cast(list[Sequence[float]], get_text_embeddings(batch))
            return [[float(x) for x in v] for v in vecs]

        return [
            [float(x) for x in cast(Sequence[float], model.get_text_embedding(t))]
            for t in batch
        ]
