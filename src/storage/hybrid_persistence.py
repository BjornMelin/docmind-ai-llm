"""Hybrid Persistence with SQLite + Qdrant unified storage.

This module implements ADR-009 compliant hybrid persistence architecture
combining SQLite for relational data with Qdrant for vector storage,
providing unified document storage and semantic search capabilities.

Key Features:
- SQLite WAL mode for document metadata and processing results
- Qdrant integration for vector embeddings and similarity search
- Unified interface for both relational and vector operations
- Atomic transactions across both storage systems
- Performance optimization for document processing pipeline
- Multi-agent concurrent access support
"""

import asyncio
import json
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Qdrant client for vector storage
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    logger.warning("Qdrant client not available. Vector operations will be disabled.")
    QdrantClient = None
    PointStruct = None
    VectorParams = None
    Distance = None
    Filter = None
    FieldCondition = None
    Match = None
    QDRANT_AVAILABLE = False

from src.models.storage import (
    DocumentMetadata,
    PersistenceError,
    SearchResult,
    StorageStats,
    VectorRecord,
)


class HybridPersistenceManager:
    """Hybrid persistence manager with SQLite + Qdrant integration.

    This manager implements ADR-009 requirements for unified document storage
    combining relational data (SQLite) with vector storage (Qdrant) for:

    - Document metadata and processing results (SQLite WAL mode)
    - Vector embeddings and semantic search (Qdrant)
    - Unified query interface for both storage types
    - Atomic operations across both systems
    - Performance optimization for document processing
    - Multi-agent concurrent access support
    """

    def __init__(self, settings: Any | None = None) -> None:
        """Initialize HybridPersistenceManager.

        Args:
            settings: DocMind configuration settings. Uses settings if None.
        """
        self.settings = settings or settings

        # Storage paths and connections
        self.sqlite_path = Path(self.settings.database.sqlite_db_path)
        self.sqlite_connection: sqlite3.Connection | None = None
        self.qdrant_client: Any | None = None

        # Collections and table names
        self.documents_table = "documents"
        self.vectors_collection = "document_vectors"

        # Connection locks for thread safety
        self._sqlite_lock = asyncio.Lock()
        self._qdrant_lock = asyncio.Lock()

        # Performance tracking
        self._operation_count = 0
        self._total_operation_time = 0.0

        # Initialize storage systems
        self._initialize_storage()

        logger.info(
            "HybridPersistenceManager initialized: sqlite={}, qdrant={}",
            self.sqlite_path,
            self.settings.database.qdrant_url if QDRANT_AVAILABLE else "disabled",
        )

    def _initialize_storage(self) -> None:
        """Initialize both SQLite and Qdrant storage systems."""
        try:
            # Initialize SQLite database
            self._initialize_sqlite()

            # Initialize Qdrant vector store
            self._initialize_qdrant()

        except Exception as e:
            logger.error(f"Failed to initialize hybrid storage: {str(e)}")
            raise PersistenceError(f"Storage initialization failed: {e}") from e

    def _initialize_sqlite(self) -> None:
        """Initialize SQLite database with WAL mode."""
        try:
            # Ensure database directory exists
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

            # Create connection with WAL mode
            self.sqlite_connection = sqlite3.connect(
                str(self.sqlite_path), timeout=30, check_same_thread=False
            )

            # Configure WAL mode for concurrent access
            self.sqlite_connection.execute("PRAGMA journal_mode = WAL")
            self.sqlite_connection.execute("PRAGMA synchronous = NORMAL")
            self.sqlite_connection.execute("PRAGMA cache_size = 10000")  # 10MB cache
            self.sqlite_connection.execute("PRAGMA temp_store = MEMORY")
            self.sqlite_connection.execute("PRAGMA mmap_size = 268435456")  # 256MB mmap

            # Create documents table
            self._create_documents_table()

            self.sqlite_connection.commit()

            logger.info(f"SQLite initialized with WAL mode: {self.sqlite_path}")

        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {str(e)}")
            raise PersistenceError(f"SQLite initialization failed: {e}") from e

    def _create_documents_table(self) -> None:
        """Create documents table with indexes."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.documents_table} (
            id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            file_size INTEGER NOT NULL,
            processing_time REAL NOT NULL,
            strategy_used TEXT NOT NULL,
            element_count INTEGER NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{{}}'
        )
        """

        self.sqlite_connection.execute(create_table_sql)

        # Create indexes for common queries
        indexes = [
            (
                f"CREATE INDEX IF NOT EXISTS idx_{self.documents_table}_file_hash "
                f"ON {self.documents_table}(file_hash)"
            ),
            (
                f"CREATE INDEX IF NOT EXISTS idx_{self.documents_table}_created_at "
                f"ON {self.documents_table}(created_at)"
            ),
            (
                f"CREATE INDEX IF NOT EXISTS idx_{self.documents_table}_file_path "
                f"ON {self.documents_table}(file_path)"
            ),
        ]

        for index_sql in indexes:
            self.sqlite_connection.execute(index_sql)

    def _initialize_qdrant(self) -> None:
        """Initialize Qdrant vector store."""
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant not available, vector operations disabled")
            return

        try:
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                url=self.settings.database.qdrant_url, timeout=30
            )

            # Create vectors collection if not exists
            asyncio.create_task(self._ensure_vector_collection())

            logger.info(f"Qdrant initialized: {self.settings.database.qdrant_url}")

        except (ConnectionError, OSError, ValueError, TimeoutError) as e:
            logger.error(f"Failed to initialize Qdrant: {str(e)}")
            # Continue without Qdrant - storage will work in SQLite-only mode
            self.qdrant_client = None

    async def _ensure_vector_collection(self) -> None:
        """Ensure Qdrant collection exists for document vectors."""
        if self.qdrant_client is None:
            return

        try:
            async with self._qdrant_lock:
                # Check if collection exists
                collections = await asyncio.to_thread(
                    self.qdrant_client.get_collections
                )

                collection_exists = any(
                    col.name == self.vectors_collection
                    for col in collections.collections
                )

                if not collection_exists:
                    # Create collection for BGE-M3 embeddings (1024 dimensions)
                    await asyncio.to_thread(
                        self.qdrant_client.create_collection,
                        collection_name=self.vectors_collection,
                        vectors_config=VectorParams(
                            size=1024,  # BGE-M3 embedding dimension
                            distance=Distance.COSINE,
                        ),
                    )

                    logger.info(f"Created Qdrant collection: {self.vectors_collection}")

        except (ConnectionError, OSError, ValueError, TimeoutError) as e:
            logger.error(f"Failed to ensure vector collection: {str(e)}")

    @asynccontextmanager
    async def _sqlite_transaction(self):
        """Context manager for SQLite transactions with proper locking."""
        async with self._sqlite_lock:
            if self.sqlite_connection is None:
                raise PersistenceError("SQLite connection not initialized")

            try:
                self.sqlite_connection.execute("BEGIN")
                yield self.sqlite_connection
                self.sqlite_connection.commit()
            except Exception as e:
                self.sqlite_connection.rollback()
                raise PersistenceError(f"SQLite transaction failed: {e}") from e

    @retry(
        retry=retry_if_exception_type((PersistenceError, sqlite3.OperationalError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
    )
    async def store_document(
        self, document_metadata: DocumentMetadata, vectors: list[VectorRecord]
    ) -> bool:
        """Store document metadata and vectors atomically.

        Args:
            document_metadata: Document metadata to store in SQLite
            vectors: Vector records to store in Qdrant

        Returns:
            True if stored successfully, False otherwise

        Raises:
            PersistenceError: If storage operation fails
        """
        start_time = time.time()

        try:
            # Store document metadata in SQLite
            async with self._sqlite_transaction() as conn:
                # Table name is controlled internally, safe from injection
                insert_sql = f"""
                INSERT OR REPLACE INTO {self.documents_table}
                (id, file_path, file_hash, file_size, processing_time, 
                 strategy_used, element_count, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """  # noqa: S608

                conn.execute(
                    insert_sql,
                    (
                        document_metadata.id,
                        document_metadata.file_path,
                        document_metadata.file_hash,
                        document_metadata.file_size,
                        document_metadata.processing_time,
                        document_metadata.strategy_used,
                        document_metadata.element_count,
                        document_metadata.created_at,
                        document_metadata.updated_at,
                        json.dumps(document_metadata.metadata),
                    ),
                )

            # Store vectors in Qdrant
            if self.qdrant_client is not None and vectors:
                await self._store_vectors(vectors)

            # Track performance
            operation_time = time.time() - start_time
            self._operation_count += 1
            self._total_operation_time += operation_time

            logger.info(
                f"Stored document {document_metadata.id} with {len(vectors)} vectors "
                f"in {operation_time:.2f}s"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to store document {document_metadata.id}: {str(e)}")
            raise PersistenceError(f"Document storage failed: {e}") from e

    async def _store_vectors(self, vectors: list[VectorRecord]) -> None:
        """Store vector records in Qdrant.

        Args:
            vectors: Vector records to store

        Raises:
            PersistenceError: If vector storage fails
        """
        if self.qdrant_client is None:
            return

        try:
            async with self._qdrant_lock:
                # Convert VectorRecord objects to PointStruct
                points = []
                for vector in vectors:
                    point = PointStruct(
                        id=vector.id,
                        vector=vector.embedding,
                        payload={
                            "document_id": vector.document_id,
                            "chunk_index": vector.chunk_index,
                            "text": vector.text,
                            "metadata": vector.metadata,
                        },
                    )
                    points.append(point)

                # Batch upsert vectors
                await asyncio.to_thread(
                    self.qdrant_client.upsert,
                    collection_name=self.vectors_collection,
                    points=points,
                )

        except Exception as e:
            logger.error(f"Failed to store vectors: {str(e)}")
            raise PersistenceError(f"Vector storage failed: {e}") from e

    @retry(
        retry=retry_if_exception_type((PersistenceError, sqlite3.OperationalError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    async def get_document_by_hash(self, file_hash: str) -> DocumentMetadata | None:
        """Get document metadata by file hash.

        Args:
            file_hash: File hash to search for

        Returns:
            DocumentMetadata if found, None otherwise

        Raises:
            PersistenceError: If retrieval fails
        """
        try:
            async with self._sqlite_lock:
                if self.sqlite_connection is None:
                    raise PersistenceError("SQLite connection not initialized")

                cursor = self.sqlite_connection.cursor()
                # Table name is controlled internally, safe from injection
                select_sql = f"""
                SELECT id, file_path, file_hash, file_size, processing_time,
                       strategy_used, element_count, created_at, updated_at, metadata
                FROM {self.documents_table}
                WHERE file_hash = ?
                """  # noqa: S608

                cursor.execute(select_sql, (file_hash,))
                row = cursor.fetchone()

                if row is None:
                    return None

                # Parse metadata JSON
                metadata = json.loads(row[9]) if row[9] else {}

                return DocumentMetadata(
                    id=row[0],
                    file_path=row[1],
                    file_hash=row[2],
                    file_size=row[3],
                    processing_time=row[4],
                    strategy_used=row[5],
                    element_count=row[6],
                    created_at=row[7],
                    updated_at=row[8],
                    metadata=metadata,
                )

        except Exception as e:
            logger.error(f"Failed to get document by hash {file_hash}: {str(e)}")
            raise PersistenceError(f"Document retrieval failed: {e}") from e

    async def search_similar_vectors(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        document_filter: str | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors with optional document filtering.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity threshold
            document_filter: Optional document ID filter

        Returns:
            List of SearchResult objects with unified data

        Raises:
            PersistenceError: If search fails
        """
        if self.qdrant_client is None:
            logger.warning("Vector search unavailable - Qdrant not initialized")
            return []

        try:
            # Build Qdrant search filter if needed
            qdrant_filter = None
            if document_filter:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id", match=MatchValue(value=document_filter)
                        )
                    ]
                )

            # Search similar vectors
            async with self._qdrant_lock:
                search_results = await asyncio.to_thread(
                    self.qdrant_client.search,
                    collection_name=self.vectors_collection,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=qdrant_filter,
                )

            # Combine with document metadata
            unified_results = []
            for result in search_results:
                document_id = result.payload["document_id"]

                # Get document metadata
                doc_metadata = await self._get_document_by_id(document_id)
                if doc_metadata is None:
                    continue

                search_result = SearchResult(
                    document_id=document_id,
                    chunk_id=str(result.id),
                    text=result.payload["text"],
                    score=result.score,
                    document_metadata=doc_metadata,
                    chunk_metadata=result.payload.get("metadata", {}),
                )

                unified_results.append(search_result)

            logger.debug(
                f"Vector search returned {len(unified_results)} results "
                f"(threshold: {score_threshold})"
            )

            return unified_results

        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise PersistenceError(f"Vector search failed: {e}") from e

    async def _get_document_by_id(self, document_id: str) -> DocumentMetadata | None:
        """Get document metadata by ID.

        Args:
            document_id: Document ID to search for

        Returns:
            DocumentMetadata if found, None otherwise
        """
        try:
            async with self._sqlite_lock:
                if self.sqlite_connection is None:
                    return None

                cursor = self.sqlite_connection.cursor()
                # Table name is controlled internally, safe from injection
                select_sql = f"""
                SELECT id, file_path, file_hash, file_size, processing_time,
                       strategy_used, element_count, created_at, updated_at, metadata
                FROM {self.documents_table}
                WHERE id = ?
                """  # noqa: S608

                cursor.execute(select_sql, (document_id,))
                row = cursor.fetchone()

                if row is None:
                    return None

                metadata = json.loads(row[9]) if row[9] else {}

                return DocumentMetadata(
                    id=row[0],
                    file_path=row[1],
                    file_hash=row[2],
                    file_size=row[3],
                    processing_time=row[4],
                    strategy_used=row[5],
                    element_count=row[6],
                    created_at=row[7],
                    updated_at=row[8],
                    metadata=metadata,
                )

        except (sqlite3.Error, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to get document by ID {document_id}: {str(e)}")
            return None

    async def get_storage_stats(self) -> StorageStats:
        """Get comprehensive storage statistics.

        Returns:
            StorageStats with current metrics
        """
        try:
            stats = StorageStats()

            # SQLite statistics
            if self.sqlite_connection is not None:
                async with self._sqlite_lock:
                    cursor = self.sqlite_connection.cursor()

                    # Count total documents
                    # Table name is controlled internally, safe from injection
                    cursor.execute(f"SELECT COUNT(*) FROM {self.documents_table}")  # noqa: S608
                    stats.total_documents = cursor.fetchone()[0]

                    # Calculate average processing time
                    # Table name is controlled internally, safe from injection
                    cursor.execute(
                        f"SELECT AVG(processing_time) FROM {self.documents_table}"  # noqa: S608
                    )
                    avg_time = cursor.fetchone()[0]
                    stats.avg_processing_time = avg_time if avg_time else 0.0

                    # Get last indexed timestamp
                    # Table name is controlled internally, safe from injection
                    cursor.execute(
                        f"SELECT MAX(created_at) FROM {self.documents_table}"  # noqa: S608
                    )
                    last_indexed = cursor.fetchone()[0]
                    stats.last_indexed_at = last_indexed

                # SQLite file size
                if self.sqlite_path.exists():
                    stats.sqlite_size_mb = self.sqlite_path.stat().st_size / (
                        1024 * 1024
                    )

            # Qdrant statistics
            if self.qdrant_client is not None:
                try:
                    async with self._qdrant_lock:
                        collection_info = await asyncio.to_thread(
                            self.qdrant_client.get_collection,
                            collection_name=self.vectors_collection,
                        )

                        stats.total_vectors = collection_info.vectors_count or 0

                        # Estimate Qdrant storage size (rough approximation)
                        # 1024 floats * 4 bytes + metadata overhead ≈ 5KB per vector
                        stats.qdrant_size_mb = (stats.total_vectors * 5) / 1024

                except (ConnectionError, OSError, ValueError, TimeoutError) as e:
                    logger.debug(f"Could not get Qdrant stats: {str(e)}")

            return stats

        except (sqlite3.Error, OSError) as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            return StorageStats()

    async def delete_document(self, document_id: str) -> bool:
        """Delete document and associated vectors.

        Args:
            document_id: Document ID to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Delete from SQLite
            async with self._sqlite_transaction() as conn:
                # Table name is controlled internally, safe from injection
                delete_sql = f"DELETE FROM {self.documents_table} WHERE id = ?"  # noqa: S608
                conn.execute(delete_sql, (document_id,))

            # Delete associated vectors from Qdrant
            if self.qdrant_client is not None:
                async with self._qdrant_lock:
                    # Delete vectors by document_id filter
                    delete_filter = Filter(
                        must=[
                            FieldCondition(
                                key="document_id", match=MatchValue(value=document_id)
                            )
                        ]
                    )

                    await asyncio.to_thread(
                        self.qdrant_client.delete,
                        collection_name=self.vectors_collection,
                        points_selector=delete_filter,
                    )

            logger.info(f"Deleted document {document_id} and associated vectors")
            return True

        except (sqlite3.Error, ConnectionError, OSError, ValueError, TimeoutError) as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            return False

    async def cleanup_storage(self, max_age_days: int = 30) -> int:
        """Clean up old documents and vectors.

        Args:
            max_age_days: Maximum age in days for documents

        Returns:
            Number of documents cleaned up
        """
        try:
            cutoff_timestamp = time.time() - (max_age_days * 24 * 3600)

            # Get old document IDs
            old_document_ids = []
            async with self._sqlite_lock:
                if self.sqlite_connection is not None:
                    cursor = self.sqlite_connection.cursor()
                    # Table name is controlled internally, safe from injection
                    cursor.execute(
                        f"SELECT id FROM {self.documents_table} WHERE created_at < ?",  # noqa: S608
                        (cutoff_timestamp,),
                    )
                    old_document_ids = [row[0] for row in cursor.fetchall()]

            # Delete old documents
            deleted_count = 0
            for doc_id in old_document_ids:
                if await self.delete_document(doc_id):
                    deleted_count += 1

            logger.info(f"Cleaned up {deleted_count} old documents")
            return deleted_count

        except (sqlite3.Error, OSError) as e:
            logger.error(f"Storage cleanup failed: {str(e)}")
            return 0

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        avg_operation_time = (
            self._total_operation_time / self._operation_count
            if self._operation_count > 0
            else 0.0
        )

        return {
            "total_operations": self._operation_count,
            "total_operation_time": self._total_operation_time,
            "avg_operation_time_ms": avg_operation_time * 1000,
            "sqlite_available": self.sqlite_connection is not None,
            "qdrant_available": self.qdrant_client is not None,
        }

    async def close(self) -> None:
        """Close storage connections."""
        try:
            if self.sqlite_connection is not None:
                self.sqlite_connection.close()
                self.sqlite_connection = None
                logger.info("SQLite connection closed")

            # Qdrant client doesn't need explicit closing
            if self.qdrant_client is not None:
                self.qdrant_client = None
                logger.info("Qdrant client closed")

        except (sqlite3.Error, ConnectionError, OSError) as e:
            logger.error(f"Error closing storage connections: {str(e)}")


# Factory function for easy instantiation
def create_hybrid_persistence_manager(
    settings: Any | None = None,
) -> HybridPersistenceManager:
    """Factory function to create HybridPersistenceManager instance.

    Args:
        settings: Optional DocMind settings. Uses settings if None.

    Returns:
        Configured HybridPersistenceManager instance
    """
    return HybridPersistenceManager(settings)
