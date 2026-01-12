"""Utilities package for DocMind AI.

This package provides small, reusable helpers used across the app, grouped into:

- `core`: async timers, startup validation, resource helpers
- `document`: document loading, cache management, spaCy bootstrapping
- `monitoring`: logging and performance monitoring utilities
- `storage`: Qdrant/vector store helpers and safe GPU contexts
"""

# Core utilities
from .core import (
    async_timer,
    detect_hardware,
    managed_async_qdrant_client,
    managed_gpu_operation,
    validate_startup_configuration,
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

# Storage and resource management operations
from .storage import (
    # Resource management operations
    async_gpu_memory_context,
    # Database operations
    clear_collection,
    create_async_client,
    create_sync_client,
    create_vector_store,
    cuda_error_context,
    get_collection_info,
    get_safe_gpu_info,
    get_safe_vram_usage,
    gpu_memory_context,
    model_context,
    safe_cuda_operation,
    setup_hybrid_collection,
    setup_hybrid_collection_async,
    sync_model_context,
    test_connection,
)

__all__ = [
    "async_gpu_memory_context",
    "async_performance_timer",
    "async_timer",
    "clear_collection",
    "clear_document_cache",
    "create_async_client",
    "create_sync_client",
    "create_vector_store",
    "cuda_error_context",
    "detect_hardware",
    "ensure_spacy_model",
    "get_cache_stats",
    "get_collection_info",
    "get_document_info",
    "get_memory_usage",
    "get_performance_monitor",
    "get_safe_gpu_info",
    "get_safe_vram_usage",
    "get_system_info",
    "gpu_memory_context",
    "load_documents_from_directory",
    "load_documents_unstructured",
    "log_error_with_context",
    "log_performance",
    "managed_async_qdrant_client",
    "managed_gpu_operation",
    "model_context",
    "performance_timer",
    "safe_cuda_operation",
    "setup_hybrid_collection",
    "setup_hybrid_collection_async",
    "setup_logging",
    "sync_model_context",
    "test_connection",
    "validate_startup_configuration",
]
