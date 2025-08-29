"""Fixed comprehensive test suite for storage systems.

This module provides working tests for the hybrid persistence system
with proper mocking and error handling.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.models.storage import (
    DocumentMetadata,
    PersistenceError,
    SearchResult,
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


class TestHybridPersistenceManagerBasic:
    """Test suite for basic HybridPersistenceManager functionality."""

    @pytest.mark.integration
    def test_initialization_sqlite_only(self, mock_settings):
        """Test successful initialization with SQLite only."""
        # Patch to avoid async task creation during initialization
        with patch("asyncio.create_task"):
            manager = HybridPersistenceManager(mock_settings)

            assert manager.settings == mock_settings
            assert manager.sqlite_path == Path(mock_settings.database.sqlite_db_path)
            assert manager.sqlite_connection is not None
            assert manager.documents_table == "documents"
            assert manager.vectors_collection == "document_vectors"

    @pytest.mark.integration
    def test_sqlite_table_creation(self, mock_settings):
        """Test that SQLite initialization creates required tables."""
        with patch("asyncio.create_task"):
            manager = HybridPersistenceManager(mock_settings)

            # Check that documents table exists
            cursor = manager.sqlite_connection.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
            )
            assert cursor.fetchone() is not None

            # Check basic schema
            cursor.execute("PRAGMA table_info(documents)")
            columns = [row[1] for row in cursor.fetchall()]
            expected_columns = ["id", "file_path", "file_hash", "file_size"]
            assert all(col in columns for col in expected_columns)

    @pytest.mark.integration
    def test_wal_mode_configuration(self, mock_settings):
        """Test that SQLite is properly configured with WAL mode."""
        with patch("asyncio.create_task"):
            manager = HybridPersistenceManager(mock_settings)

            cursor = manager.sqlite_connection.cursor()
            cursor.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]
            assert journal_mode.lower() == "wal"

    @pytest.mark.integration
    async def test_document_storage_and_retrieval(
        self, mock_settings, sample_document_metadata
    ):
        """Test basic document storage and retrieval."""
        with patch("asyncio.create_task"):
            manager = HybridPersistenceManager(mock_settings)

            # Store document (without vectors for simplicity)
            result = await manager.store_document(sample_document_metadata, [])
            assert result is True

            # Retrieve document
            retrieved = await manager.get_document_by_hash(
                sample_document_metadata.file_hash
            )
            assert retrieved is not None
            assert retrieved.id == sample_document_metadata.id
            assert retrieved.file_hash == sample_document_metadata.file_hash
            assert retrieved.metadata == sample_document_metadata.metadata

    @pytest.mark.integration
    async def test_document_not_found(self, mock_settings):
        """Test retrieval of non-existent document."""
        with patch("asyncio.create_task"):
            manager = HybridPersistenceManager(mock_settings)

            result = await manager.get_document_by_hash("nonexistent_hash")
            assert result is None

    @pytest.mark.integration
    async def test_storage_statistics(self, mock_settings, sample_document_metadata):
        """Test storage statistics functionality."""
        with patch("asyncio.create_task"):
            manager = HybridPersistenceManager(mock_settings)

            # Initially no documents
            stats = await manager.get_storage_stats()
            assert stats.total_documents == 0

            # Store a document
            await manager.store_document(sample_document_metadata, [])

            # Check updated stats
            stats = await manager.get_storage_stats()
            assert stats.total_documents == 1
            assert stats.avg_processing_time == sample_document_metadata.processing_time


class TestHybridPersistenceManagerTransactions:
    """Test suite for transaction handling."""

    @pytest.mark.integration
    async def test_transaction_context_manager(self, mock_settings):
        """Test SQLite transaction context manager."""
        with patch("asyncio.create_task"):
            manager = HybridPersistenceManager(mock_settings)

            # Test successful transaction
            async with manager._sqlite_transaction() as conn:
                conn.execute(
                    "INSERT INTO documents (id, file_path, file_hash, file_size, "
                    "processing_time, strategy_used, element_count, created_at, "
                    "updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "test_doc",
                        "/test/path.pdf",
                        "test_hash",
                        1000,
                        1.0,
                        "test",
                        10,
                        time.time(),
                        time.time(),
                        "{}",
                    ),
                )

            # Verify document was stored
            cursor = manager.sqlite_connection.cursor()
            cursor.execute("SELECT id FROM documents WHERE id = ?", ("test_doc",))
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == "test_doc"

    @pytest.mark.integration
    async def test_transaction_rollback(self, mock_settings):
        """Test transaction rollback on error."""
        with patch("asyncio.create_task"):
            manager = HybridPersistenceManager(mock_settings)

            try:
                async with manager._sqlite_transaction() as conn:
                    conn.execute(
                        "INSERT INTO documents (id, file_path, file_hash, file_size, "
                        "processing_time, strategy_used, element_count, created_at, "
                        "updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            "rollback_doc",
                            "/test/rollback.pdf",
                            "rollback_hash",
                            1000,
                            1.0,
                            "test",
                            10,
                            time.time(),
                            time.time(),
                            "{}",
                        ),
                    )
                    # Force error to trigger rollback
                    raise ValueError("Test rollback")
            except ValueError:
                pass

            # Verify document was not stored due to rollback
            cursor = manager.sqlite_connection.cursor()
            cursor.execute("SELECT id FROM documents WHERE id = ?", ("rollback_doc",))
            result = cursor.fetchone()
            assert result is None


class TestHybridPersistenceManagerVectorOperations:
    """Test suite for vector operations (with mocking)."""

    @pytest.mark.integration
    async def test_vector_search_no_qdrant(self, mock_settings):
        """Test vector search when Qdrant is not available."""
        with patch("asyncio.create_task"):
            manager = HybridPersistenceManager(mock_settings)
            # Qdrant is not available by default

            query_vector = [0.1, 0.2] * 512  # 1024 dimensions
            results = await manager.search_similar_vectors(query_vector)

            assert results == []

    @pytest.mark.integration
    async def test_vector_search_with_mocked_qdrant(
        self, mock_settings, sample_document_metadata
    ):
        """Test vector search with mocked Qdrant client."""
        with patch("asyncio.create_task"):
            manager = HybridPersistenceManager(mock_settings)

            # Mock Qdrant client
            mock_qdrant = Mock()
            manager.qdrant_client = mock_qdrant

            # Store document for metadata lookup
            await manager.store_document(sample_document_metadata, [])

            # Mock search result
            mock_search_result = Mock()
            mock_search_result.id = "vec_001"
            mock_search_result.score = 0.85
            mock_search_result.payload = {
                "document_id": sample_document_metadata.id,
                "text": "Test search result text",
                "metadata": {"page": 1},
            }

            with patch("asyncio.to_thread", return_value=[mock_search_result]):
                query_vector = [0.1, 0.2] * 512
                results = await manager.search_similar_vectors(
                    query_vector=query_vector, limit=5, score_threshold=0.7
                )

                assert len(results) == 1
                assert isinstance(results[0], SearchResult)
                assert results[0].document_id == sample_document_metadata.id
                assert results[0].score == 0.85


class TestHybridPersistenceManagerErrorHandling:
    """Test suite for error handling."""

    @pytest.mark.integration
    def test_initialization_with_invalid_path(self):
        """Test initialization with invalid database path."""
        invalid_settings = Mock()
        invalid_settings.database.sqlite_db_path = "/invalid/path/test.db"
        invalid_settings.database.qdrant_url = "http://localhost:6333"
        invalid_settings.database.qdrant_timeout = 30

        with patch("asyncio.create_task"), pytest.raises(PersistenceError):
            HybridPersistenceManager(invalid_settings)

    @pytest.mark.integration
    async def test_connection_error_handling(self, mock_settings):
        """Test handling of connection errors."""
        with patch("asyncio.create_task"):
            manager = HybridPersistenceManager(mock_settings)

            # Close connection to simulate error
            manager.sqlite_connection.close()
            manager.sqlite_connection = None

            with pytest.raises(PersistenceError):
                await manager.get_document_by_hash("test_hash")


class TestHybridPersistenceManagerPerformance:
    """Test suite for performance tracking."""

    @pytest.mark.integration
    def test_performance_stats_initialization(self, mock_settings):
        """Test performance statistics initialization."""
        with patch("asyncio.create_task"):
            manager = HybridPersistenceManager(mock_settings)

            stats = manager.get_performance_stats()
            assert stats["total_operations"] == 0
            assert stats["total_operation_time"] == 0.0
            assert stats["avg_operation_time_ms"] == 0.0
            assert stats["sqlite_available"] is True
            assert stats["qdrant_available"] is False  # Not initialized by default

    @pytest.mark.integration
    async def test_operation_time_tracking(
        self, mock_settings, sample_document_metadata
    ):
        """Test that operations are tracked for performance."""
        with patch("asyncio.create_task"):
            manager = HybridPersistenceManager(mock_settings)

            initial_stats = manager.get_performance_stats()
            initial_count = initial_stats["total_operations"]

            # Perform an operation
            await manager.store_document(sample_document_metadata, [])

            final_stats = manager.get_performance_stats()
            assert final_stats["total_operations"] == initial_count + 1
            assert final_stats["avg_operation_time_ms"] > 0


class TestHybridPersistenceManagerCleanup:
    """Test suite for resource cleanup."""

    @pytest.mark.integration
    async def test_resource_cleanup(self, mock_settings):
        """Test proper resource cleanup."""
        with patch("asyncio.create_task"):
            manager = HybridPersistenceManager(mock_settings)

            # Verify connection is active
            assert manager.sqlite_connection is not None

            await manager.close()

            # Verify connection is closed
            assert manager.sqlite_connection is None

    @pytest.mark.integration
    async def test_cleanup_old_documents(self, mock_settings):
        """Test cleanup of old documents."""
        with patch("asyncio.create_task"):
            manager = HybridPersistenceManager(mock_settings)

            # Create old document
            old_timestamp = time.time() - (40 * 24 * 3600)  # 40 days ago
            old_doc = DocumentMetadata(
                id="old_doc",
                file_path="/test/old.pdf",
                file_hash="old_hash",
                file_size=1000,
                processing_time=1.0,
                strategy_used="test",
                element_count=10,
                created_at=old_timestamp,
                updated_at=old_timestamp,
            )

            # Store old document
            await manager.store_document(old_doc, [])

            # Cleanup documents older than 30 days
            deleted_count = await manager.cleanup_storage(max_age_days=30)
            assert deleted_count == 1

            # Verify document was deleted
            result = await manager.get_document_by_hash("old_hash")
            assert result is None
