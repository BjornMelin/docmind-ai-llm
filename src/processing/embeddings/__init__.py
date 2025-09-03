"""DocMind AI BGE-M3 Embeddings Module.

This module implements BGE-M3 8K context embeddings optimized for the document
processing pipeline with unified dense/sparse embeddings, batch processing,
and semantic caching integration following ADR-009 requirements.

Components:
    bgem3_embedder: BGEM3Embedder with 8K context window and GPU optimization
    models: Embedding-specific Pydantic models

Key Features:
- BGE-M3 unified dense/sparse embeddings with 8K context window
- Optimal batch processing for document chunks
- GPU memory optimization for RTX 4090
- Semantic similarity caching integration
- Async processing with memory management
- Performance targets: <50ms per chunk, <3GB VRAM usage
"""

from src.models.embeddings import (
    EmbeddingError,
    EmbeddingParameters,
    EmbeddingResult,
)
from src.processing.embeddings.bgem3_embedder import (
    BGEM3Embedder,
)

__all__ = [
    "BGEM3Embedder",
    "EmbeddingError",
    "EmbeddingParameters",
    "EmbeddingResult",
]
