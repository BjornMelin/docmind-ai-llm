"""DocMind AI Cache Module.

This module provides dual-layer caching architecture combining LlamaIndex
IngestionCache (Layer 1) with GPTCache semantic similarity caching (Layer 2)
for 80-95% performance gains following ADR-009 requirements.

Components:
    dual_cache: DualCacheManager with IngestionCache + GPTCache integration
    models: Cache-specific Pydantic models
"""

from src.cache.dual_cache import DualCacheManager, create_dual_cache_manager
from src.cache.models import (
    CacheCoordinationResult,
    CacheError,
    CacheHitResult,
    CacheLayer,
    CacheStats,
)

__all__ = [
    "DualCacheManager",
    "create_dual_cache_manager",
    "CacheLayer",
    "CacheHitResult",
    "CacheStats",
    "CacheCoordinationResult",
    "CacheError",
]
