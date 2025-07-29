"""Utilities package for DocMind AI.

This package contains supporting utilities for logging, hardware detection,
document loading, indexing, and more.

Modules:
    document_loader: Handles document loading with support for various formats
    index_builder: Creates and manages search indices
    model_manager: Manages AI models and embeddings
    qdrant_utils: Qdrant-specific utilities
    utils: General utility functions
"""

from .document_loader import (
    create_native_multimodal_embeddings,
    extract_images_from_pdf,
    load_documents_llama,
)
from .index_builder import create_index, create_index_async
from .model_manager import ModelManager
from .qdrant_utils import (
    create_qdrant_hybrid_query,
    create_qdrant_hybrid_query_async,
    setup_hybrid_qdrant,
    setup_hybrid_qdrant_async,
)
from .utils import detect_hardware, setup_logging, verify_rrf_configuration

__all__ = [
    "setup_logging",
    "detect_hardware",
    "verify_rrf_configuration",
    "ModelManager",
    "extract_images_from_pdf",
    "create_native_multimodal_embeddings",
    "load_documents_llama",
    "setup_hybrid_qdrant_async",
    "setup_hybrid_qdrant",
    "create_qdrant_hybrid_query_async",
    "create_qdrant_hybrid_query",
    "create_index_async",
    "create_index",
]
