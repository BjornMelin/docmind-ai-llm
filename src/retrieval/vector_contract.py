"""Import-light semantic identity for DocMind vector storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config.settings import settings

if TYPE_CHECKING:
    from src.config.settings import DocMindSettings

DENSE_VECTOR_NAME = "text-dense"
SPARSE_VECTOR_NAME = "text-sparse"
SPARSE_ENCODING_CONTRACT = "bm42:doc=embed;query=query_embed:v1"


def sparse_retrieval_enabled(cfg: DocMindSettings = settings) -> bool:
    """Return whether canonical sparse vectors are required."""
    return bool(cfg.retrieval.enable_server_hybrid or cfg.retrieval.enable_keyword_tool)


__all__ = [
    "DENSE_VECTOR_NAME",
    "SPARSE_ENCODING_CONTRACT",
    "SPARSE_VECTOR_NAME",
    "sparse_retrieval_enabled",
]
