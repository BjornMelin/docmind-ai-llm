"""DocMind AI Embeddings (Library-First).

This package exposes helpers and models for the embedding pipeline, aligned
with SPEC-003:

Components:
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

__all__ = [
    "EmbeddingError",
    "EmbeddingParameters",
    "EmbeddingResult",
]
