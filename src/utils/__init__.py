"""Utilities package for DocMind AI.

This package provides small, reusable helpers used across the app, grouped into:

- `core`: async timers, startup validation, resource helpers
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
from .hashing import sha256_file
from .images import ensure_thumbnail, open_image_encrypted

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
from .security import decrypt_file, encrypt_file, get_image_kid
from .storage import (
    async_gpu_memory_context,
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
from .time import now_ms

__all__ = [
    "async_gpu_memory_context",
    "async_performance_timer",
    "async_timer",
    "clear_collection",
    "create_async_client",
    "create_sync_client",
    "create_vector_store",
    "cuda_error_context",
    "decrypt_file",
    "detect_hardware",
    "encrypt_file",
    "ensure_thumbnail",
    "get_collection_info",
    "get_image_kid",
    "get_memory_usage",
    "get_performance_monitor",
    "get_safe_gpu_info",
    "get_safe_vram_usage",
    "get_system_info",
    "gpu_memory_context",
    "log_error_with_context",
    "log_performance",
    "managed_async_qdrant_client",
    "managed_gpu_operation",
    "model_context",
    "now_ms",
    "open_image_encrypted",
    "performance_timer",
    "safe_cuda_operation",
    "setup_hybrid_collection",
    "setup_hybrid_collection_async",
    "setup_logging",
    "sha256_file",
    "sync_model_context",
    "test_connection",
    "validate_startup_configuration",
]
