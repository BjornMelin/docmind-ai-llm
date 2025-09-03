"""Unit tests for HybridPersistenceManager storage operations.

Covers document storage, vector search, error handling, and performance stats
with boundary testing and business logic validation.

Note: Some internal helpers are exercised where no public surface exists yet
to validate error paths and transactions; usage is narrowly scoped.
"""

# pylint: disable=protected-access

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.storage import (
    DocumentMetadata,
    PersistenceError,
    SearchResult,
    StorageStats,
    VectorRecord,
)
from src.storage.hybrid_persistence import HybridPersistenceManager


class TestHybridPersistenceInitialization:
    """Test HybridPersistenceManager initialization."""

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    def test_init_sqlite_only_mode(self):
        """Test initialization with SQLite only (Qdrant disabled)."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        assert manager.sqlite_connection is not None
        assert manager.qdrant_client is None
        assert manager.documents_table == "documents"
        assert manager.vectors_collection == "document_vectors"
        # No internal counters; verify core structures are initialized
        assert isinstance(manager.documents_table, str)

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    def test_init_with_qdrant(self, mock_qdrant_client):
        """Test initialization with Qdrant enabled."""
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        # Avoid scheduling async task without a running loop
        with patch("asyncio.create_task") as _noop:
            manager = HybridPersistenceManager(mock_settings)

        assert manager.sqlite_connection is not None
        assert manager.qdrant_client == mock_client

    def test_init_sqlite_file_creation(self):
        """Test SQLite database file creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"

            mock_settings = MagicMock()
            mock_settings.database.sqlite_db_path = str(db_path)
            mock_settings.database.qdrant_url = "http://localhost:6333"

            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager = HybridPersistenceManager(mock_settings)

            # Database file should be created
            assert db_path.exists()
            assert manager.sqlite_connection is not None

    def test_init_sqlite_error(self):
        """Test initialization with SQLite error."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = "/invalid/path/test.db"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        with (
            patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False),
            pytest.raises(PersistenceError),  # Should raise PersistenceError
        ):
            HybridPersistenceManager(mock_settings)


class TestStoreDocumentCritical:
    """Test the critical store_document function."""

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_store_document_sqlite_only(self):
        """Test storing document with SQLite only."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        # Create test document metadata
        doc_metadata = DocumentMetadata(
            id="test-doc-1",
            file_path="/test/document.pdf",
            file_hash="abc123hash",
            file_size=1024,
            processing_time=1.5,
            strategy_used="unstructured",
            element_count=10,
            created_at=1234567890.0,
            updated_at=1234567890.0,
            metadata={"test": "metadata"},
        )

        # Test vectors (will be ignored in SQLite-only mode)
        test_vectors = [
            VectorRecord(
                id="vec-1",
                document_id="test-doc-1",
                chunk_index=0,
                text="Test content",
                embedding=[0.1] * 1024,
                metadata={"chunk": 0},
            )
        ]

        result = await manager.store_document(doc_metadata, test_vectors)

        assert result
        # Verify SQLite insert occurred by checking cursor execution count via
        # connection mockability. As a proxy in unit scope, confirm the call
        # returned True and manager still connected
        assert manager.sqlite_connection is not None

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_store_document_with_vectors(self, mock_qdrant_client):
        """Test storing document with vectors in Qdrant."""
        mock_client = MagicMock()
        mock_client.upsert = MagicMock()
        mock_qdrant_client.return_value = mock_client

        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        doc_metadata = DocumentMetadata(
            id="test-doc-1",
            file_path="/test/document.pdf",
            file_hash="abc123hash",
            file_size=1024,
            processing_time=1.5,
            strategy_used="unstructured",
            element_count=10,
            created_at=1234567890.0,
            updated_at=1234567890.0,
            metadata={},
        )

        test_vectors = [
            VectorRecord(
                id="vec-1",
                document_id="test-doc-1",
                chunk_index=0,
                text="Test content",
                embedding=[0.1] * 1024,
                metadata={"chunk": 0},
            )
        ]

        result = await manager.store_document(doc_metadata, test_vectors)

        assert result
        # Should have called Qdrant upsert
        assert mock_client.upsert.called

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_store_document_duplicate_handling(self):
        """Test storing document with duplicate file hash."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        doc_metadata1 = DocumentMetadata(
            id="test-doc-1",
            file_path="/test/document.pdf",
            file_hash="same-hash",
            file_size=1024,
            processing_time=1.5,
            strategy_used="unstructured",
            element_count=10,
            created_at=1234567890.0,
            updated_at=1234567890.0,
            metadata={},
        )

        doc_metadata2 = DocumentMetadata(
            id="test-doc-2",
            file_path="/test/document2.pdf",
            file_hash="same-hash",  # Same hash - should replace
            file_size=2048,
            processing_time=2.0,
            strategy_used="unstructured",
            element_count=15,
            created_at=1234567891.0,
            updated_at=1234567891.0,
            metadata={},
        )

        # Store first document
        result1 = await manager.store_document(doc_metadata1, [])
        assert result1

        # Store second document with same hash
        result2 = await manager.store_document(doc_metadata2, [])
        assert result2

        # Should have replaced the first document
        retrieved = await manager.get_document_by_hash("same-hash")
        assert retrieved is not None
        assert retrieved.id == "test-doc-2"
        assert retrieved.file_size == 2048

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_store_document_empty_vectors(self):
        """Test storing document with empty vectors list."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        doc_metadata = DocumentMetadata(
            id="test-doc-1",
            file_path="/test/document.pdf",
            file_hash="abc123hash",
            file_size=1024,
            processing_time=1.5,
            strategy_used="unstructured",
            element_count=10,
            created_at=1234567890.0,
            updated_at=1234567890.0,
            metadata={},
        )

        result = await manager.store_document(doc_metadata, [])

        assert result

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_store_document_error_handling(self):
        """Test store_document error handling."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        # Close the connection to simulate error
        manager.sqlite_connection.close()
        manager.sqlite_connection = None

        doc_metadata = DocumentMetadata(
            id="test-doc-1",
            file_path="/test/document.pdf",
            file_hash="abc123hash",
            file_size=1024,
            processing_time=1.5,
            strategy_used="unstructured",
            element_count=10,
            created_at=1234567890.0,
            updated_at=1234567890.0,
            metadata={},
        )

        with pytest.raises(PersistenceError):  # Should raise PersistenceError
            await manager.store_document(doc_metadata, [])


class TestSearchSimilarVectorsCritical:
    """Test the critical search_similar_vectors function."""

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_search_similar_vectors_qdrant_disabled(self):
        """Test vector search when Qdrant is disabled."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        query_vector = [0.1] * 1024
        results = await manager.search_similar_vectors(query_vector)

        # Should return empty list when Qdrant disabled
        assert results == []

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_search_similar_vectors_success(self, mock_qdrant_client):
        """Test successful vector similarity search."""
        # Mock Qdrant search results
        mock_search_result = MagicMock()
        mock_search_result.id = "vec-1"
        mock_search_result.score = 0.95
        mock_search_result.payload = {
            "document_id": "doc-1",
            "chunk_index": 0,
            "text": "Test content",
            "metadata": {"chunk": 0},
        }

        mock_client = MagicMock()
        mock_client.search = MagicMock(return_value=[mock_search_result])
        mock_qdrant_client.return_value = mock_client

        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        # First store a document so we have metadata
        doc_metadata = DocumentMetadata(
            id="doc-1",
            file_path="/test/document.pdf",
            file_hash="hash123",
            file_size=1024,
            processing_time=1.5,
            strategy_used="unstructured",
            element_count=10,
            created_at=1234567890.0,
            updated_at=1234567890.0,
            metadata={},
        )
        await manager.store_document(doc_metadata, [])

        # Mock async call
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = [mock_search_result]

            query_vector = [0.1] * 1024
            results = await manager.search_similar_vectors(
                query_vector, limit=5, score_threshold=0.8
            )

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].document_id == "doc-1"
        assert results[0].score == 0.95
        assert results[0].text == "Test content"

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_search_similar_vectors_with_filter(self, mock_qdrant_client):
        """Test vector search with document filter."""
        mock_client = MagicMock()
        mock_client.search = MagicMock(return_value=[])
        mock_qdrant_client.return_value = mock_client

        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = []

            query_vector = [0.1] * 1024
            results = await manager.search_similar_vectors(
                query_vector, document_filter="specific-doc-id"
            )

        assert results == []
        # Should have called search with filter
        mock_to_thread.assert_called_once()

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_search_similar_vectors_no_metadata(self, mock_qdrant_client):
        """Test vector search when document metadata not found."""
        mock_search_result = MagicMock()
        mock_search_result.id = "vec-1"
        mock_search_result.score = 0.95
        mock_search_result.payload = {
            "document_id": "nonexistent-doc",
            "chunk_index": 0,
            "text": "Test content",
            "metadata": {},
        }

        mock_client = MagicMock()
        mock_client.search = MagicMock(return_value=[mock_search_result])
        mock_qdrant_client.return_value = mock_client

        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = [mock_search_result]

            query_vector = [0.1] * 1024
            results = await manager.search_similar_vectors(query_vector)

        # Should skip results where document metadata not found
        assert results == []

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_search_similar_vectors_error_handling(self, mock_qdrant_client):
        """Test vector search error handling."""
        mock_client = MagicMock()
        mock_client.search = MagicMock(side_effect=ConnectionError("Qdrant error"))
        mock_qdrant_client.return_value = mock_client

        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = ConnectionError("Qdrant error")

            query_vector = [0.1] * 1024

            with pytest.raises(PersistenceError):  # Should raise PersistenceError
                await manager.search_similar_vectors(query_vector)


class TestDocumentRetrieval:
    """Test document retrieval functions."""

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_get_document_by_hash_success(self):
        """Test successful document retrieval by hash."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        # Store a document first
        doc_metadata = DocumentMetadata(
            id="test-doc-1",
            file_path="/test/document.pdf",
            file_hash="target-hash",
            file_size=1024,
            processing_time=1.5,
            strategy_used="unstructured",
            element_count=10,
            created_at=1234567890.0,
            updated_at=1234567890.0,
            metadata={"test": "data"},
        )
        await manager.store_document(doc_metadata, [])

        # Retrieve by hash
        retrieved = await manager.get_document_by_hash("target-hash")

        assert retrieved is not None
        assert retrieved.id == "test-doc-1"
        assert retrieved.file_hash == "target-hash"
        assert retrieved.file_size == 1024
        assert retrieved.metadata["test"] == "data"

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_get_document_by_hash_not_found(self):
        """Test document retrieval when hash not found."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        result = await manager.get_document_by_hash("nonexistent-hash")

        assert result is None

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_get_document_by_id_success(self):
        """Test successful document retrieval by ID."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        # Store a document first
        doc_metadata = DocumentMetadata(
            id="target-id",
            file_path="/test/document.pdf",
            file_hash="some-hash",
            file_size=1024,
            processing_time=1.5,
            strategy_used="unstructured",
            element_count=10,
            created_at=1234567890.0,
            updated_at=1234567890.0,
            metadata={},
        )
        await manager.store_document(doc_metadata, [])

        # Retrieve by ID
        retrieved = await manager._get_document_by_id("target-id")

        assert retrieved is not None
        assert retrieved.id == "target-id"
        assert retrieved.file_hash == "some-hash"


class TestStorageStatistics:
    """Test storage statistics and performance tracking."""

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_get_storage_stats_sqlite_only(self):
        """Test storage statistics with SQLite only."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        # Store some test documents
        for i in range(3):
            doc_metadata = DocumentMetadata(
                id=f"doc-{i}",
                file_path=f"/test/doc{i}.pdf",
                file_hash=f"hash-{i}",
                file_size=1024 * (i + 1),
                processing_time=1.0 + i * 0.5,
                strategy_used="unstructured",
                element_count=10 + i,
                created_at=1234567890.0 + i,
                updated_at=1234567890.0 + i,
                metadata={},
            )
            await manager.store_document(doc_metadata, [])

        stats = await manager.get_storage_stats()

        assert isinstance(stats, StorageStats)
        assert stats.total_documents == 3
        assert stats.total_vectors == 0  # Qdrant disabled
        assert stats.avg_processing_time == 1.5  # (1.0 + 1.5 + 2.0) / 3
        assert stats.last_indexed_at == 1234567892.0  # Last document timestamp
        assert stats.sqlite_size_mb >= 0
        assert stats.qdrant_size_mb == 0

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_get_storage_stats_with_qdrant(self, mock_qdrant_client):
        """Test storage statistics with Qdrant enabled."""
        # Mock Qdrant collection info
        mock_collection_info = MagicMock()
        mock_collection_info.vectors_count = 150

        mock_client = MagicMock()
        mock_client.get_collection = MagicMock(return_value=mock_collection_info)
        mock_qdrant_client.return_value = mock_client

        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_collection_info

            stats = await manager.get_storage_stats()

        assert stats.total_vectors == 150
        assert stats.qdrant_size_mb > 0  # Should estimate size

    @pytest.mark.asyncio
    async def test_get_performance_stats_basic(self):
        """Test basic performance statistics."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
            manager = HybridPersistenceManager(mock_settings)

        # Prefer black‑box stats: invoke operations and then fetch stats
        import time as _time

        start = _time.perf_counter()
        # Perform a couple of lightweight operations that update counters
        from src.models.storage import DocumentMetadata

        doc1 = DocumentMetadata(
            id="perf-1",
            file_path="/perf/doc1.pdf",
            file_hash="perf-hash-1",
            file_size=1,
            processing_time=0.1,
            strategy_used="unstructured",
            element_count=1,
            created_at=0.0,
            updated_at=0.0,
            metadata={},
        )
        doc2 = DocumentMetadata(
            id="perf-2",
            file_path="/perf/doc2.pdf",
            file_hash="perf-hash-2",
            file_size=1,
            processing_time=0.2,
            strategy_used="unstructured",
            element_count=1,
            created_at=0.0,
            updated_at=0.0,
            metadata={},
        )
        await manager.store_document(doc1, [])
        await manager.store_document(doc2, [])
        elapsed_ms = (_time.perf_counter() - start) * 1000

        stats = manager.get_performance_stats()

        assert isinstance(stats, dict)
        assert stats["total_operations"] >= 2
        assert stats["avg_operation_time_ms"] >= 0.0
        assert stats["avg_operation_time_ms"] <= max(elapsed_ms, 1000.0)
        assert stats["sqlite_available"]
        assert not stats["qdrant_available"]

    def test_get_performance_stats_no_operations(self):
        """Test performance statistics with no operations."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
            manager = HybridPersistenceManager(mock_settings)

        stats = manager.get_performance_stats()

        assert stats["total_operations"] == 0
        assert stats["avg_operation_time_ms"] == 0.0


class TestStorageManagement:
    """Test storage management functions."""

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_delete_document_success(self):
        """Test successful document deletion."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        # Store a document first
        doc_metadata = DocumentMetadata(
            id="doc-to-delete",
            file_path="/test/document.pdf",
            file_hash="delete-hash",
            file_size=1024,
            processing_time=1.5,
            strategy_used="unstructured",
            element_count=10,
            created_at=1234567890.0,
            updated_at=1234567890.0,
            metadata={},
        )
        await manager.store_document(doc_metadata, [])

        # Verify it exists
        retrieved = await manager.get_document_by_hash("delete-hash")
        assert retrieved is not None

        # Delete it
        result = await manager.delete_document("doc-to-delete")
        assert result

        # Verify it's gone
        retrieved_after = await manager.get_document_by_hash("delete-hash")
        assert retrieved_after is None

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_cleanup_storage_by_age(self):
        """Test storage cleanup by document age."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        import time

        current_time = time.time()
        old_time = current_time - (31 * 24 * 3600)  # 31 days ago
        recent_time = current_time - (1 * 24 * 3600)  # 1 day ago

        # Store old and recent documents
        old_doc = DocumentMetadata(
            id="old-doc",
            file_path="/test/old.pdf",
            file_hash="old-hash",
            file_size=1024,
            processing_time=1.5,
            strategy_used="unstructured",
            element_count=10,
            created_at=old_time,
            updated_at=old_time,
            metadata={},
        )

        recent_doc = DocumentMetadata(
            id="recent-doc",
            file_path="/test/recent.pdf",
            file_hash="recent-hash",
            file_size=1024,
            processing_time=1.5,
            strategy_used="unstructured",
            element_count=10,
            created_at=recent_time,
            updated_at=recent_time,
            metadata={},
        )

        await manager.store_document(old_doc, [])
        await manager.store_document(recent_doc, [])

        # Cleanup documents older than 30 days
        deleted_count = await manager.cleanup_storage(max_age_days=30)

        assert deleted_count == 1  # Only old document should be deleted

        # Verify recent document still exists
        recent_retrieved = await manager.get_document_by_hash("recent-hash")
        assert recent_retrieved is not None

        # Verify old document is gone
        old_retrieved = await manager.get_document_by_hash("old-hash")
        assert old_retrieved is None

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_close_connections(self):
        """Test closing storage connections."""
        mock_settings = MagicMock()
        mock_settings.database.sqlite_db_path = ":memory:"
        mock_settings.database.qdrant_url = "http://localhost:6333"

        manager = HybridPersistenceManager(mock_settings)

        # Verify connections are open
        assert manager.sqlite_connection is not None

        # Close connections
        await manager.close()

        # Verify connections are closed
        assert manager.sqlite_connection is None
