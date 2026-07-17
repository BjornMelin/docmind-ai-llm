"""Canonical pinned BM42 document and query encoding for Qdrant."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import cache
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from qdrant_client import models as qmodels

from src.config import settings
from src.config.embedding_defaults import (
    DEFAULT_BM42_FILES,
    DEFAULT_BM42_MODEL_ID,
    DEFAULT_BM42_SOURCE_REPO,
    DEFAULT_BM42_SOURCE_REVISION,
)

SparseBatch = tuple[list[list[int]], list[list[float]]]
SparseEncoder = Callable[[list[str]], SparseBatch]


class SparseEncodingError(RuntimeError):
    """Raised when the canonical sparse model cannot encode a complete batch."""


@cache
def _get_sparse_encoder(cache_folder: str) -> Any:
    """Load the immutable BM42 snapshot from the configured offline cache."""
    from fastembed import SparseTextEmbedding

    snapshot_path = snapshot_download(
        repo_id=DEFAULT_BM42_SOURCE_REPO,
        revision=DEFAULT_BM42_SOURCE_REVISION,
        allow_patterns=list(DEFAULT_BM42_FILES),
        cache_dir=cache_folder,
        local_files_only=True,
    )
    return SparseTextEmbedding(
        DEFAULT_BM42_MODEL_ID,
        cache_dir=cache_folder,
        specific_model_path=snapshot_path,
        local_files_only=True,
        providers=["CPUExecutionProvider"],
    )


def _cache_folder() -> str:
    """Return one normalized model-cache owner for every sparse call site."""
    return str(Path(settings.embedding.cache_folder).expanduser().resolve())


def _encode(texts: Iterable[str], *, query: bool) -> SparseBatch:
    """Encode a complete batch with the canonical document or query algorithm."""
    batch = [str(text) for text in texts]
    if not batch:
        return [], []
    encoder = _get_sparse_encoder(_cache_folder())
    embeddings = list(encoder.query_embed(batch) if query else encoder.embed(batch))
    if len(embeddings) != len(batch):
        raise SparseEncodingError(
            "Canonical sparse encoder returned an incomplete batch"
        )

    indices_batch: list[list[int]] = []
    values_batch: list[list[float]] = []
    for embedding in embeddings:
        indices = [int(value) for value in getattr(embedding, "indices", ())]
        values = [float(value) for value in getattr(embedding, "values", ())]
        if len(indices) != len(values):
            raise SparseEncodingError(
                "Canonical sparse encoder returned mismatched indices and values"
            )
        indices_batch.append(indices)
        values_batch.append(values)
    return indices_batch, values_batch


def encode_documents(texts: list[str]) -> SparseBatch:
    """Encode document chunks with BM42 attention-weighted document semantics."""
    return _encode(texts, query=False)


def encode_queries(texts: list[str]) -> SparseBatch:
    """Encode queries with BM42's dedicated unit-weight query semantics."""
    return _encode(texts, query=True)


def sparse_callbacks() -> tuple[SparseEncoder, SparseEncoder]:
    """Return explicit callbacks for LlamaIndex's Qdrant adapter."""
    return encode_documents, encode_queries


def encode_to_qdrant(text: str) -> qmodels.SparseVector | None:
    """Encode one query using the same canonical callback as every retriever."""
    try:
        indices_batch, values_batch = encode_queries([text])
    except SparseEncodingError:
        raise
    except Exception as exc:
        raise SparseEncodingError("Canonical sparse query encoding failed") from exc
    indices = indices_batch[0]
    values = values_batch[0]
    if not indices:
        return None
    return qmodels.SparseVector(indices=indices, values=values)


__all__ = [
    "SparseEncodingError",
    "encode_documents",
    "encode_queries",
    "encode_to_qdrant",
    "sparse_callbacks",
]
