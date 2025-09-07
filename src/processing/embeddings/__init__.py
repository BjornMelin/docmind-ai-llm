"""DocMind AI Embeddings (Library-First).

This package exposes helpers and models for the embedding pipeline, aligned
with SPEC-003:

Components:
- LlamaIndex BGEM3Index/BGEM3Retriever (tri-mode text: dense+sparse+ColBERT)
- LlamaIndex ClipEmbedding for image vectors in LI contexts
- Pydantic models for embedding parameters and results

Key Principles:
- Prefer library implementations over custom wrappers
- Derive image dimensions at runtime (no hard-coded OpenCLIP/SigLIP dims)
- Keep tests fast/offline via stubs/monkeypatching
"""

from src.models.embeddings import (
    EmbeddingError,
    EmbeddingParameters,
    EmbeddingResult,
)
from src.retrieval.bge_m3_index import build_bge_m3_index, build_bge_m3_retriever

__all__ = [
    "EmbeddingError",
    "EmbeddingParameters",
    "EmbeddingResult",
    "build_bge_m3_index",
    "build_bge_m3_retriever",
]
