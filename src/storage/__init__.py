"""DocMind AI Storage Module.

This module provides hybrid persistence architecture combining Qdrant vector
storage with SQLite relational storage for comprehensive data management
and retrieval capabilities following ADR-009 requirements.

Components:
    hybrid_persistence: HybridPersistenceManager with Qdrant + SQLite integration
    models: Storage-specific Pydantic models

Key Features:
- Qdrant vector storage for embeddings and similarity search
- SQLite relational storage for metadata and session persistence
- Hybrid persistence layer for unified data access
- WAL mode for concurrent access optimization
"""

from src.models.storage import (
    DocumentMetadata,
    PersistenceError,
    SearchResult,
    StorageStats,
    VectorRecord,
)
from src.storage.hybrid_persistence import HybridPersistenceManager

__all__ = [
    "DocumentMetadata",
    "HybridPersistenceManager",
    "PersistenceError",
    "SearchResult",
    "StorageStats",
    "VectorRecord",
]
