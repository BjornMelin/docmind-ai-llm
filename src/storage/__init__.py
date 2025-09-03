"""DocMind AI Storage Module.

Library-first storage utilities. Vector storage is provided via
LlamaIndex QdrantVectorStore and StorageContext (see src.utils.storage).
Operational metadata may use SQLite via application components when needed.
"""

from src.models.storage import (
    DocumentMetadata,
    PersistenceError,
    SearchResult,
    StorageStats,
    VectorRecord,
)

__all__ = [
    "DocumentMetadata",
    "PersistenceError",
    "SearchResult",
    "StorageStats",
    "VectorRecord",
]
