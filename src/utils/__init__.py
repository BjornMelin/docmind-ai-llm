"""Consolidated utilities package for DocMind AI.

This package contains essential utilities in 5 consolidated modules:
- core: Essential hardware detection and validation utilities
- document: Document loading and spaCy model management
- embedding: Embedding model creation and basic indexing
- database: Qdrant client management and hybrid collection setup
- monitoring: Simple performance monitoring and logging

The package follows KISS, DRY, YAGNI principles with library-first implementations.
"""  # noqa: N999

# Core utilities
from .core import (
    async_timer,
    detect_hardware,
    managed_async_qdrant_client,
    managed_gpu_operation,
    validate_startup_configuration,
    verify_rrf_configuration,
)

# Database operations
from .database import (
    clear_collection,
    create_async_client,
    create_sync_client,
    create_vector_store,
    get_collection_info,
    setup_hybrid_collection,
    setup_hybrid_collection_async,
    test_connection,
)

# Document operations
from .document import (
    clear_document_cache,
    ensure_spacy_model,
    get_cache_stats,
    get_document_info,
    load_documents_from_directory,
    load_documents_unstructured,
)

# Embedding operations - removed (moved to src.retrieval)
# The embedding module was deprecated and replaced by src.retrieval.integration
# Monitoring and logging
from .monitoring import (
    async_performance_timer,
    get_memory_usage,
    get_performance_monitor,
    get_system_info,
    log_error_with_context,
    log_performance,
    performance_timer,
    setup_logging,
)

__all__ = [
    # Core utilities
    "detect_hardware",
    "verify_rrf_configuration",
    "validate_startup_configuration",
    "async_timer",
    "managed_gpu_operation",
    "managed_async_qdrant_client",
    # Document operations
    "load_documents_unstructured",
    "load_documents_from_directory",
    "get_document_info",
    "clear_document_cache",
    "get_cache_stats",
    "ensure_spacy_model",
    # Database operations
    "create_sync_client",
    "create_async_client",
    "setup_hybrid_collection",
    "setup_hybrid_collection_async",
    "create_vector_store",
    "get_collection_info",
    "test_connection",
    "clear_collection",
    # Monitoring and logging
    "setup_logging",
    "log_error_with_context",
    "log_performance",
    "performance_timer",
    "async_performance_timer",
    "get_memory_usage",
    "get_system_info",
    "get_performance_monitor",
]
