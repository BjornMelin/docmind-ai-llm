"""Storage-specific Pydantic models for DocMind AI Hybrid Persistence System.

This module contains data models specifically designed for the hybrid persistence
architecture combining SQLite document metadata storage with Qdrant vector storage,
following requirements for unified document storage and efficient retrieval.

Models:
    DocumentMetadata: Document metadata stored in SQLite
    VectorRecord: Vector record stored in Qdrant
    SearchResult: Unified search result combining relational and vector data
    StorageStats: Storage system statistics
    PersistenceError: Custom exception for persistence operation errors
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    """Document metadata stored in SQLite."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(description="Unique document ID")
    file_path: str = Field(description="Original file path")
    file_hash: str = Field(description="Document content hash")
    file_size: int = Field(description="File size in bytes")
    processing_time: float = Field(description="Processing time in seconds")
    strategy_used: str = Field(description="Processing strategy applied")
    element_count: int = Field(description="Number of extracted elements")
    created_at: float = Field(description="Creation timestamp")
    updated_at: float = Field(description="Last update timestamp")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class VectorRecord(BaseModel):
    """Vector record stored in Qdrant."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(description="Unique vector ID")
    document_id: str = Field(description="Associated document ID")
    chunk_index: int = Field(description="Chunk index within document")
    text: str = Field(description="Original text content")
    embedding: list[float] = Field(description="Vector embedding")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Vector metadata"
    )


class SearchResult(BaseModel):
    """Unified search result combining relational and vector data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    document_id: str = Field(description="Document ID")
    chunk_id: str = Field(description="Chunk/vector ID")
    text: str = Field(description="Matching text content")
    score: float = Field(description="Similarity score")
    document_metadata: DocumentMetadata = Field(description="Document metadata")
    chunk_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Chunk metadata"
    )


class StorageStats(BaseModel):
    """Storage system statistics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_documents: int = Field(default=0, description="Total documents stored")
    total_vectors: int = Field(default=0, description="Total vectors stored")
    sqlite_size_mb: float = Field(default=0.0, description="SQLite database size in MB")
    qdrant_size_mb: float = Field(default=0.0, description="Qdrant storage size in MB")
    avg_processing_time: float = Field(
        default=0.0, description="Average processing time"
    )
    last_indexed_at: float | None = Field(
        default=None, description="Last indexing timestamp"
    )


class PersistenceError(Exception):
    """Custom exception for persistence operation errors."""
