"""DocMind AI Document Processing Module.

This module implements ADR-009 compliant document processing with direct
Unstructured.io integration, semantic chunking, and BGE-M3 embeddings
for the document processing pipeline.

Components:
    resilient_processor: ResilientDocumentProcessor with direct Unstructured.io
    chunking: Semantic chunking with chunk_by_title implementation
    embeddings: BGE-M3 8K context embeddings for document processing
    models: Processing-specific Pydantic models

Key Features:
- Direct unstructured.partition.auto.partition() calls
- Semantic chunking with chunk_by_title()
- BGE-M3 unified dense/sparse embeddings
- 8K context window support
- Async processing with error resilience
"""

from src.processing.embeddings.models import (
    EmbeddingError,
    EmbeddingParameters,
    EmbeddingResult,
)
from src.processing.models import (
    DocumentElement,
    ProcessingError,
    ProcessingResult,
    ProcessingStrategy,
)
from src.processing.resilient_processor import (
    ResilientDocumentProcessor,
    create_resilient_processor,
)

__all__ = [
    "ResilientDocumentProcessor",
    "create_resilient_processor",
    "ProcessingStrategy",
    "DocumentElement",
    "ProcessingResult",
    "ProcessingError",
    "EmbeddingParameters",
    "EmbeddingResult",
    "EmbeddingError",
]
