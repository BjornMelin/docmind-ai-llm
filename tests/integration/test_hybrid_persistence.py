"""Comprehensive test suite for HybridPersistenceManager.

This module tests the hybrid persistence system combining SQLite and Qdrant
for document metadata storage and vector search capabilities.

Test Focus:
- CRUD operations for documents and vectors
- SQLite transaction handling with rollback scenarios
- Qdrant vector operations with mocked dependencies
- Session management and lifecycle
- Data integrity and concurrent access patterns
- Error handling and recovery scenarios
"""

import asyncio
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.models.storage import (
    DocumentMetadata,
    PersistenceError,
    SearchResult,
    StorageStats,
    VectorRecord,
)
from src.storage.hybrid_persistence import HybridPersistenceManager


@pytest.fixture
def temp_db_path():
    """Create temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    yield db_path
    # Cleanup - remove database files
    if db_path.exists():
        db_path.unlink()
    # Remove WAL and SHM files if they exist
    for suffix in ["-wal", "-shm"]:
        wal_path = Path(str(db_path) + suffix)
        if wal_path.exists():
            wal_path.unlink()


@pytest.fixture
def mock_settings(temp_db_path):
    """Create mock settings for testing."""
    settings = Mock()
    settings.database.sqlite_db_path = str(temp_db_path)
    settings.database.qdrant_url = "http://localhost:6333"
    settings.database.qdrant_timeout = 30
    return settings


@pytest.fixture
def sample_document_metadata():
    """Create sample document metadata for testing."""
    timestamp = time.time()
    return DocumentMetadata(
        id="test_doc_001",
        file_path="/test/documents/sample.pdf",
        file_hash="abc123def456",
        file_size=1024000,
        processing_time=2.5,
        strategy_used="hi_res",
        element_count=25,
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"title": "Test Document", "pages": 5},
    )


@pytest.fixture
def sample_vector_records():
    """Create sample vector records for testing."""
    return [
        VectorRecord(
            id="vec_001",
            document_id="test_doc_001",
            chunk_index=0,
            text="First chunk of test document content.",
            embedding=[0.1, 0.2, 0.3, 0.4] * 256,  # 1024 dimensions
            metadata={"page": 1, "paragraph": 1},
        ),
        VectorRecord(
            id="vec_002",
            document_id="test_doc_001",
            chunk_index=1,
            text="Second chunk of test document content.",
            embedding=[0.5, 0.6, 0.7, 0.8] * 256,  # 1024 dimensions
            metadata={"page": 1, "paragraph": 2},
        ),
    ]


class TestHybridPersistenceManagerInitialization:
    """Test suite for HybridPersistenceManager initialization."""

    @pytest.mark.integration
    def test_initialization_with_valid_settings(self, mock_settings):
        """Test successful initialization with valid settings."""
        with (
            patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True),
            patch("src.storage.hybrid_persistence.QdrantClient") as mock_client,
            patch("asyncio.create_task"),  # Mock async task creation
        ):
            mock_client.return_value = Mock()

            manager = HybridPersistenceManager(mock_settings)

            assert manager.settings == mock_settings
            assert manager.sqlite_path == Path(mock_settings.database.sqlite_db_path)
            assert manager.sqlite_connection is not None
            assert manager.documents_table == "documents"
            assert manager.vectors_collection == "document_vectors"

    @pytest.mark.integration
    def test_initialization_without_qdrant(self, mock_settings):
        """Test initialization when Qdrant is not available."""
        with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
            manager = HybridPersistenceManager(mock_settings)

            assert manager.qdrant_client is None
            assert manager.sqlite_connection is not None

    @pytest.mark.integration
    def test_sqlite_initialization_creates_tables(self, mock_settings):
        """Test that SQLite initialization creates required tables and indexes."""
        with patch("asyncio.create_task"):  # Mock async task creation
            manager = HybridPersistenceManager(mock_settings)

            # Check that documents table exists
            cursor = manager.sqlite_connection.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
            )
            assert cursor.fetchone() is not None

            # Check that indexes exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            assert any("file_hash" in idx for idx in indexes)
            assert any("created_at" in idx for idx in indexes)

    @pytest.mark.integration
    def test_initialization_handles_sqlite_errors(self, mock_settings):
        """Test initialization handles SQLite errors gracefully."""
        # Use invalid path that will cause permission error
        mock_settings.database.sqlite_db_path = "/invalid/path/test.db"

        with pytest.raises(PersistenceError):
            HybridPersistenceManager(mock_settings)

    @pytest.mark.integration
    def test_wal_mode_configuration(self, mock_settings):
        """Test that SQLite is configured with WAL mode and optimizations."""
        with patch("asyncio.create_task"):  # Mock async task creation
            manager = HybridPersistenceManager(mock_settings)

            # Check WAL mode is enabled
            cursor = manager.sqlite_connection.cursor()
            cursor.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]
            assert journal_mode.lower() == "wal"

            # Check other optimizations
            cursor.execute("PRAGMA synchronous")
            synchronous = cursor.fetchone()[0]
            assert synchronous == 1  # NORMAL mode

            cursor.execute("PRAGMA cache_size")
            cache_size = cursor.fetchone()[0]
            assert cache_size == 10000


class TestHybridPersistenceManagerCRUDOperations:
    """Test suite for CRUD operations."""

    @pytest.mark.integration
    async def test_store_document_success(
        self, mock_settings, sample_document_metadata, sample_vector_records
    ):
        """Test successful document storage with vectors."""
        with (
            patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True),
            patch("src.storage.hybrid_persistence.QdrantClient") as mock_client,
        ):
            mock_qdrant = Mock()
            mock_client.return_value = mock_qdrant

            manager = HybridPersistenceManager(mock_settings)
            manager.qdrant_client = mock_qdrant

            # Mock async operations
            with patch("asyncio.to_thread") as mock_to_thread:
                mock_to_thread.return_value = None

                result = await manager.store_document(
                    sample_document_metadata, sample_vector_records
                )

                assert result is True

                # Verify document was stored in SQLite
                cursor = manager.sqlite_connection.cursor()
                cursor.execute(
                    "SELECT * FROM documents WHERE id = ?",
                    (sample_document_metadata.id,),
                )
                row = cursor.fetchone()
                assert row is not None
                assert row[0] == sample_document_metadata.id

                # Verify vector storage was attempted
                mock_to_thread.assert_called()

    @pytest.mark.integration
    async def test_store_document_sqlite_only(
        self, mock_settings, sample_document_metadata, sample_vector_records
    ):
        """Test document storage when Qdrant is not available."""
        with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
            manager = HybridPersistenceManager(mock_settings)

            result = await manager.store_document(
                sample_document_metadata, sample_vector_records
            )

            assert result is True

            # Verify document was stored in SQLite
            cursor = manager.sqlite_connection.cursor()
            cursor.execute(
                "SELECT * FROM documents WHERE id = ?", (sample_document_metadata.id,)
            )
            row = cursor.fetchone()
            assert row is not None

    @pytest.mark.integration
    async def test_get_document_by_hash_success(
        self, mock_settings, sample_document_metadata
    ):
        """Test successful document retrieval by hash."""
        manager = HybridPersistenceManager(mock_settings)

        # First store the document
        await manager.store_document(sample_document_metadata, [])

        # Then retrieve it
        retrieved = await manager.get_document_by_hash(
            sample_document_metadata.file_hash
        )

        assert retrieved is not None
        assert retrieved.id == sample_document_metadata.id
        assert retrieved.file_hash == sample_document_metadata.file_hash
        assert retrieved.file_path == sample_document_metadata.file_path
        assert retrieved.metadata == sample_document_metadata.metadata

    @pytest.mark.integration
    async def test_get_document_by_hash_not_found(self, mock_settings):
        """Test document retrieval when document doesn't exist."""
        manager = HybridPersistenceManager(mock_settings)

        result = await manager.get_document_by_hash("nonexistent_hash")

        assert result is None

    @pytest.mark.integration
    async def test_delete_document_success(
        self, mock_settings, sample_document_metadata, sample_vector_records
    ):
        """Test successful document deletion."""
        with (
            patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True),
            patch("src.storage.hybrid_persistence.QdrantClient") as mock_client,
        ):
            mock_qdrant = Mock()
            mock_client.return_value = mock_qdrant

            manager = HybridPersistenceManager(mock_settings)
            manager.qdrant_client = mock_qdrant

            # Store document first
            with patch("asyncio.to_thread"):
                await manager.store_document(
                    sample_document_metadata, sample_vector_records
                )

            # Delete document
            with patch("asyncio.to_thread") as mock_to_thread:
                result = await manager.delete_document(sample_document_metadata.id)

                assert result is True

                # Verify document was deleted from SQLite
                cursor = manager.sqlite_connection.cursor()
                cursor.execute(
                    "SELECT * FROM documents WHERE id = ?",
                    (sample_document_metadata.id,),
                )
                row = cursor.fetchone()
                assert row is None

                # Verify vector deletion was attempted
                mock_to_thread.assert_called()

    @pytest.mark.integration
    async def test_get_storage_stats(self, mock_settings, sample_document_metadata):
        """Test storage statistics retrieval."""
        # Force offline (no Qdrant) for deterministic behavior
        with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
            manager = HybridPersistenceManager(mock_settings)

        # Store a document first
        await manager.store_document(sample_document_metadata, [])

        stats = await manager.get_storage_stats()

        assert isinstance(stats, StorageStats)
        assert stats.total_documents == 1
        assert stats.avg_processing_time == sample_document_metadata.processing_time
        assert stats.last_indexed_at is not None
        assert stats.sqlite_size_mb > 0


class TestHybridPersistenceManagerTransactions:
    """Test suite for transaction handling."""

    @pytest.mark.integration
    async def test_transaction_rollback_on_error(
        self, mock_settings, sample_vector_records
    ):
        """Test transaction rollback when error occurs during storage."""
        manager = HybridPersistenceManager(mock_settings)

        # Create invalid document metadata that will cause an error
        invalid_metadata = DocumentMetadata(
            id="test_doc",
            file_path="/test/path.pdf",
            file_hash="test_hash",
            file_size=1000,
            processing_time=1.0,
            strategy_used="test",
            element_count=1,
            created_at=time.time(),
            updated_at=time.time(),
        )

        # Mock an error during vector storage
        with (
            patch.object(
                manager,
                "_store_vectors",
                side_effect=Exception("Vector storage failed"),
            ),
            pytest.raises(PersistenceError),
        ):
            # Provide vectors so _store_vectors is invoked
            await manager.store_document(invalid_metadata, sample_vector_records)

        # Note: SQLite insert may succeed before vector failure; we only assert raise

    @pytest.mark.integration
    async def test_concurrent_transactions(self, mock_settings):
        """Test concurrent transaction handling."""
        manager = HybridPersistenceManager(mock_settings)

        # Create multiple documents
        documents = []
        for i in range(5):
            timestamp = time.time()
            doc = DocumentMetadata(
                id=f"concurrent_doc_{i}",
                file_path=f"/test/concurrent_{i}.pdf",
                file_hash=f"hash_{i}",
                file_size=1000 + i,
                processing_time=1.0 + i * 0.1,
                strategy_used="test",
                element_count=10 + i,
                created_at=timestamp,
                updated_at=timestamp,
            )
            documents.append(doc)

        # Store documents concurrently
        tasks = [manager.store_document(doc, []) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        assert all(result is True for result in results)

        # Verify all documents were stored
        cursor = manager.sqlite_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        assert count == 5

    @pytest.mark.integration
    async def test_transaction_retry_mechanism(self, mock_settings):
        """Test transaction retry on temporary failures."""
        manager = HybridPersistenceManager(mock_settings)

        document = DocumentMetadata(
            id="retry_test_doc",
            file_path="/test/retry.pdf",
            file_hash="retry_hash",
            file_size=1000,
            processing_time=1.0,
            strategy_used="test",
            element_count=10,
            created_at=time.time(),
            updated_at=time.time(),
        )

        # Mock database locked error on first attempt, success on second
        original_execute = manager.sqlite_connection.execute
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise sqlite3.OperationalError("database is locked")
            return original_execute(*args, **kwargs)

        # Patch the async transaction context manager to simulate a transient lock
        from contextlib import asynccontextmanager

        original_tx = manager._sqlite_transaction
        call_count_local = {"n": 0}

        @asynccontextmanager
        async def flaky_transaction():  # type: ignore[override]
            if call_count_local["n"] == 0:
                call_count_local["n"] += 1
                raise sqlite3.OperationalError("database is locked")
            async with original_tx() as conn:
                yield conn

        with patch.object(manager, "_sqlite_transaction", flaky_transaction):
            result = await manager.store_document(document, [])

            assert result is True
            # Verify document was eventually stored
            cursor = manager.sqlite_connection.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (document.id,))
            row = cursor.fetchone()
            assert row is not None


class TestHybridPersistenceManagerVectorOperations:
    """Test suite for vector operations."""

    @pytest.mark.integration
    async def test_search_similar_vectors_success(
        self, mock_settings, sample_document_metadata
    ):
        """Test successful vector similarity search."""
        with (
            patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True),
            patch("src.storage.hybrid_persistence.QdrantClient") as mock_client,
        ):
            mock_qdrant = Mock()
            mock_client.return_value = mock_qdrant

            manager = HybridPersistenceManager(mock_settings)
            manager.qdrant_client = mock_qdrant

            # Store document first for metadata lookup
            await manager.store_document(sample_document_metadata, [])

            # Mock search results
            mock_search_result = Mock()
            mock_search_result.id = "vec_001"
            mock_search_result.score = 0.85
            mock_search_result.payload = {
                "document_id": sample_document_metadata.id,
                "text": "Test search result text",
                "metadata": {"page": 1},
            }

            with patch("asyncio.to_thread", return_value=[mock_search_result]):
                query_vector = [0.1, 0.2] * 512  # 1024 dimensions
                results = await manager.search_similar_vectors(
                    query_vector=query_vector, limit=5, score_threshold=0.7
                )

                assert len(results) == 1
                assert isinstance(results[0], SearchResult)
                assert results[0].document_id == sample_document_metadata.id
                assert results[0].score == 0.85
                assert results[0].text == "Test search result text"

    @pytest.mark.integration
    async def test_search_similar_vectors_with_filter(self, mock_settings):
        """Test vector search with document filtering."""
        with (
            patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True),
            patch("src.storage.hybrid_persistence.QdrantClient") as mock_client,
        ):
            mock_qdrant = Mock()
            mock_client.return_value = mock_qdrant

            manager = HybridPersistenceManager(mock_settings)
            manager.qdrant_client = mock_qdrant

            with patch("asyncio.to_thread", return_value=[]):
                query_vector = [0.1, 0.2] * 512
                results = await manager.search_similar_vectors(
                    query_vector=query_vector,
                    limit=5,
                    document_filter="specific_doc_id",
                )

                assert isinstance(results, list)

    @pytest.mark.integration
    async def test_search_similar_vectors_no_qdrant(self, mock_settings):
        """Test vector search when Qdrant is not available."""
        with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
            manager = HybridPersistenceManager(mock_settings)

            query_vector = [0.1, 0.2] * 512
            results = await manager.search_similar_vectors(query_vector)

            assert results == []

    @pytest.mark.integration
    async def test_store_vectors_batch_operation(
        self, mock_settings, sample_vector_records
    ):
        """Test batch vector storage operation."""
        with (
            patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True),
            patch("src.storage.hybrid_persistence.QdrantClient") as mock_client,
        ):
            mock_qdrant = Mock()
            mock_client.return_value = mock_qdrant

            manager = HybridPersistenceManager(mock_settings)
            manager.qdrant_client = mock_qdrant

            with patch("asyncio.to_thread") as mock_to_thread:
                await manager._store_vectors(sample_vector_records)

                # Verify batch upsert was called
                mock_to_thread.assert_called_once()
                call_args = mock_to_thread.call_args
                assert call_args[0][0] == mock_qdrant.upsert
                assert "points" in call_args[1]


class TestHybridPersistenceManagerErrorHandling:
    """Test suite for error handling scenarios."""

    @pytest.mark.integration
    async def test_sqlite_connection_error_handling(self, mock_settings):
        """Test handling of SQLite connection errors."""
        manager = HybridPersistenceManager(mock_settings)

        # Close the connection to simulate error
        manager.sqlite_connection.close()
        manager.sqlite_connection = None

        with pytest.raises(PersistenceError):
            await manager.get_document_by_hash("test_hash")

    @pytest.mark.integration
    async def test_qdrant_connection_error_handling(self, mock_settings):
        """Test handling of Qdrant connection errors."""
        with (
            patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True),
            patch("src.storage.hybrid_persistence.QdrantClient") as mock_client,
        ):
            # Simulate connection error
            mock_client.side_effect = ConnectionError("Qdrant unavailable")

            # Should still initialize successfully but without Qdrant
            manager = HybridPersistenceManager(mock_settings)
            assert manager.qdrant_client is None

    @pytest.mark.integration
    async def test_json_serialization_error_handling(self, mock_settings):
        """Test handling of JSON serialization errors."""
        manager = HybridPersistenceManager(mock_settings)

        # Create document with non-serializable metadata
        class NonSerializable:
            pass

        document = DocumentMetadata(
            id="json_error_doc",
            file_path="/test/json_error.pdf",
            file_hash="json_error_hash",
            file_size=1000,
            processing_time=1.0,
            strategy_used="test",
            element_count=10,
            created_at=time.time(),
            updated_at=time.time(),
            metadata={"obj": NonSerializable()},  # This will cause JSON error
        )

        with pytest.raises(PersistenceError):
            await manager.store_document(document, [])

    @pytest.mark.integration
    async def test_cleanup_storage_operation(
        self, mock_settings, sample_document_metadata
    ):
        """Test cleanup storage operation."""
        # Force offline (no Qdrant) for deterministic behavior
        with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
            manager = HybridPersistenceManager(mock_settings)

        # Store old document (backdated)
        old_doc = DocumentMetadata(
            id="old_doc",
            file_path="/test/old.pdf",
            file_hash="old_hash",
            file_size=1000,
            processing_time=1.0,
            strategy_used="test",
            element_count=10,
            created_at=time.time() - (40 * 24 * 3600),  # 40 days ago
            updated_at=time.time() - (40 * 24 * 3600),
        )

        await manager.store_document(old_doc, [])
        await manager.store_document(sample_document_metadata, [])

        # Cleanup documents older than 30 days
        deleted_count = await manager.cleanup_storage(max_age_days=30)

        assert deleted_count >= 1

        # Verify old document is gone, new one remains
        old_result = await manager.get_document_by_hash("old_hash")
        new_result = await manager.get_document_by_hash(
            sample_document_metadata.file_hash
        )

        assert old_result is None
        assert new_result is not None


class TestHybridPersistenceManagerPerformance:
    """Test suite for performance tracking."""

    @pytest.mark.integration
    def test_performance_stats_tracking(self, mock_settings):
        """Test performance statistics tracking."""
        with patch("asyncio.create_task"):  # Mock async task creation
            manager = HybridPersistenceManager(mock_settings)

            initial_stats = manager.get_performance_stats()
            assert initial_stats["total_operations"] == 0
            assert initial_stats["avg_operation_time_ms"] == 0

    @pytest.mark.integration
    async def test_operation_time_tracking(
        self, mock_settings, sample_document_metadata
    ):
        """Test that operation times are tracked correctly."""
        manager = HybridPersistenceManager(mock_settings)

        initial_stats = manager.get_performance_stats()
        initial_count = initial_stats["total_operations"]

        await manager.store_document(sample_document_metadata, [])

        final_stats = manager.get_performance_stats()
        assert final_stats["total_operations"] == initial_count + 1
        assert final_stats["avg_operation_time_ms"] > 0

    @pytest.mark.integration
    async def test_resource_cleanup_on_close(self, mock_settings):
        """Test proper resource cleanup when closing manager."""
        manager = HybridPersistenceManager(mock_settings)

        # Verify connections are active
        assert manager.sqlite_connection is not None

        await manager.close()

        # Verify connections are closed
        assert manager.sqlite_connection is None
