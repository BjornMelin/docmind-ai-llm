"""DocMind AI Document Processing Module.

This module implements ADR-009 compliant document processing with a unified
approach that combines unstructured.io parsing with LlamaIndex
IngestionPipeline orchestration.

Components:
    document_processor: DocumentProcessor combining unstructured + LlamaIndex
    chunking: Semantic chunking with chunk_by_title implementation
    embeddings: BGE-M3 8K context embeddings for document processing
    models: Processing-specific Pydantic models

Key Features:
- Unified approach: unstructured.io parsing + LlamaIndex pipeline orchestration
- Built-in caching, async processing, and transformations via LlamaIndex
- Strategy mapping (hi_res, fast, ocr_only) based on file type
- Semantic chunking with SentenceSplitter
- BGE-M3 unified dense/sparse embeddings support
- Performance target: >1 page/second with hi_res strategy
"""

# Import document processor and factory functions
from src.processing.document_processor import (
    DocumentProcessor,
    create_document_processor,
    create_resilient_processor,
)
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

__all__ = [
    # Primary implementation
    "DocumentProcessor",
    # Factory functions
    "create_document_processor",
    "create_resilient_processor",  # Compatibility factory
    # Essential models
    "ProcessingStrategy",
    "DocumentElement",
    "ProcessingResult",
    "ProcessingError",
    # Embedding models
    "EmbeddingParameters",
    "EmbeddingResult",
    "EmbeddingError",
]
