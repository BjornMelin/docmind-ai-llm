"""Sparse query encoding utilities for Qdrant hybrid search (library-first).

This module provides a tiny wrapper that produces a Qdrant-compatible
``SparseVector`` for a query string using FastEmbed's sparse models.

Policy:
- Prefer BM42 (attentions) when available; fall back to BM25.
- If FastEmbed is not installed or any error occurs, return ``None`` so
  callers can fall back to dense-only hybrid queries and log a telemetry flag.

Notes:
- Implementation is intentionally minimal and lazy-imports FastEmbed to
  preserve offline determinism and avoid heavy imports when unused.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from qdrant_client import models as qmodels

from src.utils.storage import FALLBACK_SPARSE_MODEL, PREFERRED_SPARSE_MODEL


@cache
def _get_sparse_encoder() -> Any | None:
    """Return a cached FastEmbed sparse encoder instance or ``None``.

    The function tries BM42 first and falls back to BM25. Errors are swallowed
    to keep the app resilient; callers should handle ``None`` by skipping
    sparse prefetch.
    """
    try:
        from fastembed import SparseTextEmbedding  # type: ignore

        try:
            return SparseTextEmbedding(PREFERRED_SPARSE_MODEL)
        except (
            RuntimeError,
            ValueError,
            OSError,
            TypeError,
        ):  # pragma: no cover - fallback path
            return SparseTextEmbedding(FALLBACK_SPARSE_MODEL)
    except ImportError:  # pragma: no cover - optional dependency
        return None


def encode_to_qdrant(text: str) -> qmodels.SparseVector | None:
    """Encode a query string to a Qdrant ``SparseVector`` using FastEmbed.

    Args:
        text: Query text to encode.

    Returns:
        qmodels.SparseVector or ``None`` when sparse encoding is unavailable.
    """
    enc = _get_sparse_encoder()
    if enc is None:
        return None

    try:
        it = enc.embed([text])
        emb = next(iter(it))
        indices = [int(i) for i in getattr(emb, "indices", [])]
        values = [float(v) for v in getattr(emb, "values", [])]
        if not indices or not values:
            return None
        return qmodels.SparseVector(indices=indices, values=values)
    except (
        StopIteration,
        AttributeError,
        TypeError,
        ValueError,
    ):  # pragma: no cover - defensive
        return None


__all__ = ["encode_to_qdrant"]
