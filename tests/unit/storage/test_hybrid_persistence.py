"""Comprehensive tests for HybridPersistenceManager missing coverage areas.

This module targets specific methods and edge cases not covered in the existing tests
to improve coverage for hybrid_persistence.py from 81.52% to a higher percentage.

Focus areas:
- Async context managers and transactions
- Vector storage edge cases
- Collection initialization edge cases
- Error recovery patterns
- Cleanup and maintenance operations
- Performance tracking edge cases
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.models.storage import (
    DocumentMetadata,
    PersistenceError,
    StorageStats,
    VectorRecord,
)


@pytest.fixture
def mock_settings():
    """Mock settings for HybridPersistenceManager."""
    settings = Mock()
    settings.database.sqlite_db_path = ":memory:"
    settings.database.qdrant_url = "http://localhost:6333"
    return settings


@pytest.fixture
def sample_document_metadata():
    """Sample DocumentMetadata for testing."""
    return DocumentMetadata(
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


@pytest.fixture
def sample_vector_records():
    """Sample VectorRecord list for testing."""
    return [
        VectorRecord(
            id="vec-1",
            document_id="test-doc-1",
            chunk_index=0,
            text="Test content 1",
            embedding=[0.1] * 1024,
            metadata={"chunk": 0},
        ),
        VectorRecord(
            id="vec-2",
            document_id="test-doc-1",
            chunk_index=1,
            text="Test content 2",
            embedding=[0.2] * 1024,
            metadata={"chunk": 1},
        ),
    ]


@pytest.mark.unit
class TestHybridPersistenceAsyncOperations:
    """Test async operations and context managers."""

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @patch("src.storage.hybrid_persistence.asyncio.create_task")
    @pytest.mark.asyncio
    async def test_ensure_vector_collection_success(
        self, mock_create_task, mock_qdrant_client, mock_settings
    ):
        """Test successful vector collection creation."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        # Mock collections response - no existing collection
        collections_response = Mock()
        collections_response.collections = []

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        # Mock async operations
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = [
                collections_response,  # get_collections
                None,  # create_collection
            ]

            await manager._ensure_vector_collection()

        # Verify calls
        assert mock_to_thread.call_count == 2

        # First call should be get_collections
        first_call = mock_to_thread.call_args_list[0]
        assert first_call[0][0] == mock_client.get_collections

        # Second call should be create_collection
        second_call = mock_to_thread.call_args_list[1]
        assert second_call[0][0] == mock_client.create_collection

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_ensure_vector_collection_existing(
        self, mock_qdrant_client, mock_settings
    ):
        """Test skipping creation when collection exists."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        # Mock collections response - existing collection
        existing_collection = Mock()
        existing_collection.name = "document_vectors"
        collections_response = Mock()
        collections_response.collections = [existing_collection]

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = collections_response

            await manager._ensure_vector_collection()

        # Should only call get_collections, not create_collection
        assert mock_to_thread.call_count == 1

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_ensure_vector_collection_error_handling(
        self, mock_qdrant_client, mock_settings
    ):
        """Test error handling during collection creation."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = ConnectionError("Qdrant connection failed")

            # Should not raise exception, just log error
            await manager._ensure_vector_collection()

        # Error should be handled gracefully
        assert mock_to_thread.call_count == 1

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_ensure_vector_collection_timeout_error(
        self, mock_qdrant_client, mock_settings
    ):
        """Test timeout error handling during collection operations."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = TimeoutError("Qdrant timeout")

            await manager._ensure_vector_collection()

        assert mock_to_thread.call_count == 1

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_ensure_vector_collection_value_error(
        self, mock_qdrant_client, mock_settings
    ):
        """Test value error handling during collection operations."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = ValueError("Invalid configuration")

            await manager._ensure_vector_collection()

        assert mock_to_thread.call_count == 1


@pytest.mark.unit
class TestHybridPersistenceVectorOperations:
    """Test vector storage operations and edge cases."""

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_store_vectors_success(
        self, mock_qdrant_client, mock_settings, sample_vector_records
    ):
        """Test successful vector storage."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = None  # Successful upsert

            await manager._store_vectors(sample_vector_records)

        # Verify upsert was called
        mock_to_thread.assert_called_once()
        upsert_call = mock_to_thread.call_args[0]
        assert upsert_call[0] == mock_client.upsert

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_store_vectors_no_client(
        self, mock_qdrant_client, mock_settings, sample_vector_records
    ):
        """Test vector storage when client is None."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_qdrant_client.return_value = None

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = None

        # Should return without error when no client
        await manager._store_vectors(sample_vector_records)

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_store_vectors_error_handling(
        self, mock_qdrant_client, mock_settings, sample_vector_records
    ):
        """Test error handling during vector storage."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = Exception("Vector storage failed")

            with pytest.raises(PersistenceError, match="Vector storage failed"):
                await manager._store_vectors(sample_vector_records)

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_store_vectors_point_conversion(
        self, mock_qdrant_client, mock_settings, sample_vector_records
    ):
        """Test proper conversion of VectorRecord to PointStruct."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = None

            await manager._store_vectors(sample_vector_records)

        # Verify the upsert call structure
        call_args = mock_to_thread.call_args
        assert call_args[1]["collection_name"] == "document_vectors"

        points = call_args[1]["points"]
        assert len(points) == 2

        # Verify point structure
        first_point = points[0]
        assert first_point.id == "vec-1"
        assert len(first_point.vector) == 1024
        assert first_point.payload["document_id"] == "test-doc-1"
        assert first_point.payload["chunk_index"] == 0
        assert first_point.payload["text"] == "Test content 1"


@pytest.mark.unit
class TestHybridPersistenceSearchOperations:
    """Test search operations and edge cases."""

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_search_similar_vectors_no_document_metadata(
        self, mock_qdrant_client, mock_settings
    ):
        """Test vector search when document metadata is not found."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        # Mock search result
        mock_result = Mock()
        mock_result.payload = {
            "document_id": "nonexistent-doc",
            "text": "Test content",
            "metadata": {},
        }

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = [mock_result]

            query_vector = [0.1] * 1024
            results = await manager.search_similar_vectors(query_vector)

        # Should return empty list when document metadata not found
        assert results == []

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_search_similar_vectors_with_document_filter(
        self, mock_qdrant_client, mock_settings
    ):
        """Test vector search with document filtering."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        # Patch Qdrant filter types to trivial dummies to avoid typing.Union issues
        class _DummyFilter:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class _DummyFieldCondition:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class _DummyMatch:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        with (
            patch("src.storage.hybrid_persistence.Filter", _DummyFilter),
            patch(
                "src.storage.hybrid_persistence.FieldCondition", _DummyFieldCondition
            ),
            patch("src.storage.hybrid_persistence.MatchValue", _DummyMatch),
            patch("asyncio.to_thread") as mock_to_thread,
        ):
            mock_to_thread.return_value = []

            query_vector = [0.1] * 1024
            await manager.search_similar_vectors(
                query_vector, document_filter="specific-doc-id"
            )

        # Verify filter was applied
        search_call = mock_to_thread.call_args
        assert search_call[1]["query_filter"] is not None


@pytest.mark.unit
class TestHybridPersistenceDocumentOperations:
    """Test document operations edge cases."""

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_get_document_by_id_no_connection(self, mock_settings):
        """Test document retrieval when SQLite connection is None."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        manager = HybridPersistenceManager(mock_settings)
        manager.sqlite_connection = None

        result = await manager._get_document_by_id("test-id")

        assert result is None

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_get_document_by_id_json_decode_error(self, mock_settings):
        """Test document retrieval with JSON decode error."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        manager = HybridPersistenceManager(mock_settings)

        # Mock sqlite connection and cursor with invalid JSON
        cursor = Mock()
        cursor.fetchone.return_value = (
            "test-id",
            "/test/path",
            "hash",
            1024,
            1.5,
            "strategy",
            10,
            123456.0,
            123457.0,
            "invalid json",
        )

        manager.sqlite_connection = Mock()
        manager.sqlite_connection.cursor.return_value = cursor

        result = await manager._get_document_by_id("test-id")

        assert result is None

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_get_document_by_id_sqlite_error(self, mock_settings):
        """Test document retrieval with SQLite error."""
        import sqlite3

        from src.storage.hybrid_persistence import HybridPersistenceManager

        manager = HybridPersistenceManager(mock_settings)

        # Mock cursor with SQLite error
        cursor = Mock()
        cursor.execute.side_effect = sqlite3.Error("Database error")
        manager.sqlite_connection = Mock()
        manager.sqlite_connection.cursor.return_value = cursor

        result = await manager._get_document_by_id("test-id")

        assert result is None

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_get_document_by_id_value_error(self, mock_settings):
        """Test document retrieval with value error."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        manager = HybridPersistenceManager(mock_settings)

        # Mock cursor that raises ValueError
        cursor = Mock()
        cursor.execute.side_effect = ValueError("Invalid data")
        manager.sqlite_connection = Mock()
        manager.sqlite_connection.cursor.return_value = cursor

        result = await manager._get_document_by_id("test-id")

        assert result is None


@pytest.mark.unit
class TestHybridPersistenceStatistics:
    """Test storage statistics and performance tracking."""

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_get_storage_stats_qdrant_error(
        self, mock_qdrant_client, mock_settings
    ):
        """Test storage statistics when Qdrant operations fail."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = ConnectionError("Qdrant error")

            stats = await manager.get_storage_stats()

        # Should handle error gracefully and return stats without Qdrant data
        assert isinstance(stats, StorageStats)
        assert stats.total_vectors == 0
        assert stats.qdrant_size_mb == 0

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_get_storage_stats_qdrant_timeout(
        self, mock_qdrant_client, mock_settings
    ):
        """Test storage statistics with Qdrant timeout."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = TimeoutError("Qdrant timeout")

            stats = await manager.get_storage_stats()

        assert isinstance(stats, StorageStats)
        assert stats.total_vectors == 0

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_get_storage_stats_qdrant_value_error(
        self, mock_qdrant_client, mock_settings
    ):
        """Test storage statistics with Qdrant value error."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = ValueError("Invalid request")

            stats = await manager.get_storage_stats()

        assert isinstance(stats, StorageStats)

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_get_storage_stats_sqlite_error(self, mock_settings):
        """Test storage statistics with SQLite error."""
        import sqlite3

        from src.storage.hybrid_persistence import HybridPersistenceManager

        manager = HybridPersistenceManager(mock_settings)

        # Mock SQLite error
        cursor = Mock()
        cursor.execute.side_effect = sqlite3.Error("Database error")
        manager.sqlite_connection = Mock()
        manager.sqlite_connection.cursor.return_value = cursor

        stats = await manager.get_storage_stats()

        # Should return empty stats on error
        assert isinstance(stats, StorageStats)
        assert stats.total_documents == 0

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_get_storage_stats_file_not_exists(self, mock_settings):
        """Test storage statistics when SQLite file doesn't exist."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        # Create manager normally, then override sqlite_path to a non-existent location
        manager = HybridPersistenceManager(mock_settings)
        manager.sqlite_path = Path("/definitely_not_exists_dir__/file.db")

        stats = await manager.get_storage_stats()

        # Should handle missing file gracefully
        assert isinstance(stats, StorageStats)
        assert stats.sqlite_size_mb == 0


@pytest.mark.unit
class TestHybridPersistenceCleanupOperations:
    """Test cleanup and maintenance operations."""

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_delete_document_qdrant_error(
        self, mock_qdrant_client, mock_settings
    ):
        """Test document deletion with Qdrant error."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        # Patch Qdrant filter classes to simple dummies
        class _DummyFilter:
            def __init__(self, *args, **kwargs):
                pass

        class _DummyFieldCondition:
            def __init__(self, *args, **kwargs):
                pass

        class _DummyMatch:
            def __init__(self, *args, **kwargs):
                pass

        with (
            patch("src.storage.hybrid_persistence.Filter", _DummyFilter),
            patch(
                "src.storage.hybrid_persistence.FieldCondition", _DummyFieldCondition
            ),
            patch("src.storage.hybrid_persistence.MatchValue", _DummyMatch),
            patch("asyncio.to_thread") as mock_to_thread,
        ):
            mock_to_thread.side_effect = ConnectionError("Qdrant delete failed")

            result = await manager.delete_document("test-doc")

        # Should return False on error
        assert result is False

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_delete_document_qdrant_timeout(
        self, mock_qdrant_client, mock_settings
    ):
        """Test document deletion with Qdrant timeout."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        class _DummyFilter:
            def __init__(self, *args, **kwargs):
                pass

        class _DummyFieldCondition:
            def __init__(self, *args, **kwargs):
                pass

        class _DummyMatch:
            def __init__(self, *args, **kwargs):
                pass

        with (
            patch("src.storage.hybrid_persistence.Filter", _DummyFilter),
            patch(
                "src.storage.hybrid_persistence.FieldCondition", _DummyFieldCondition
            ),
            patch("src.storage.hybrid_persistence.MatchValue", _DummyMatch),
            patch("asyncio.to_thread") as mock_to_thread,
        ):
            mock_to_thread.side_effect = TimeoutError("Qdrant timeout")

            result = await manager.delete_document("test-doc")

        assert result is False

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_delete_document_qdrant_value_error(
        self, mock_qdrant_client, mock_settings
    ):
        """Test document deletion with Qdrant value error."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        class _DummyFilter:
            def __init__(self, *args, **kwargs):
                pass

        class _DummyFieldCondition:
            def __init__(self, *args, **kwargs):
                pass

        class _DummyMatch:
            def __init__(self, *args, **kwargs):
                pass

        with (
            patch("src.storage.hybrid_persistence.Filter", _DummyFilter),
            patch(
                "src.storage.hybrid_persistence.FieldCondition", _DummyFieldCondition
            ),
            patch("src.storage.hybrid_persistence.MatchValue", _DummyMatch),
            patch("asyncio.to_thread") as mock_to_thread,
        ):
            mock_to_thread.side_effect = ValueError("Invalid delete request")

            result = await manager.delete_document("test-doc")

        assert result is False

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_cleanup_storage_sqlite_error(self, mock_settings):
        """Test storage cleanup with SQLite error."""
        import sqlite3

        from src.storage.hybrid_persistence import HybridPersistenceManager

        manager = HybridPersistenceManager(mock_settings)

        # Mock SQLite error
        cursor = Mock()
        cursor.execute.side_effect = sqlite3.Error("Database error")
        manager.sqlite_connection = Mock()
        manager.sqlite_connection.cursor.return_value = cursor

        result = await manager.cleanup_storage(max_age_days=30)

        # Should return 0 on error
        assert result == 0

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_cleanup_storage_os_error(self, mock_settings):
        """Test storage cleanup with OS error."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        manager = HybridPersistenceManager(mock_settings)

        # Mock OS error
        cursor = Mock()
        cursor.execute.side_effect = OSError("Disk error")
        manager.sqlite_connection = Mock()
        manager.sqlite_connection.cursor.return_value = cursor

        result = await manager.cleanup_storage(max_age_days=30)

        assert result == 0


@pytest.mark.unit
class TestHybridPersistenceConnectionManagement:
    """Test connection management and closing operations."""

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_close_sqlite_error(self, mock_qdrant_client, mock_settings):
        """Test connection closing with SQLite error."""
        import sqlite3

        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        # Mock SQLite close error
        manager.sqlite_connection = Mock()
        manager.sqlite_connection.close.side_effect = sqlite3.Error("Close error")

        # Should not raise exception
        await manager.close()

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_close_connection_error(self, mock_qdrant_client, mock_settings):
        """Test connection closing with connection error."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        # Mock connection error
        manager.sqlite_connection = Mock()
        manager.sqlite_connection.close.side_effect = ConnectionError("Network error")

        await manager.close()

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_close_os_error(self, mock_qdrant_client, mock_settings):
        """Test connection closing with OS error."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        # Mock OS error
        manager.sqlite_connection = Mock()
        manager.sqlite_connection.close.side_effect = OSError("File system error")

        await manager.close()


@pytest.mark.unit
class TestHybridPersistenceFactoryFunction:
    """Test factory function for creating manager instances."""

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    def test_create_hybrid_persistence_manager_default_settings(self, mock_settings):
        """Test factory function with default settings."""
        from src.storage.hybrid_persistence import create_hybrid_persistence_manager

        manager = create_hybrid_persistence_manager(mock_settings)

        assert manager is not None
        assert manager.settings == mock_settings

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    def test_create_hybrid_persistence_manager_none_settings(self):
        """Ensure passing None settings raises a clear error (no back-compat)."""
        from src.storage.hybrid_persistence import create_hybrid_persistence_manager

        with pytest.raises(Exception, match="invalid|None|manager|persistence"):
            create_hybrid_persistence_manager(None)


@pytest.mark.unit
class TestHybridPersistenceEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_sqlite_transaction_context_manager_error(self, mock_settings):
        """Test SQLite transaction context manager with connection error."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        manager = HybridPersistenceManager(mock_settings)
        manager.sqlite_connection = None

        # Should raise PersistenceError when connection is None
        with pytest.raises(PersistenceError, match="SQLite connection not initialized"):
            async with manager._sqlite_transaction() as _:
                pass

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_sqlite_transaction_rollback_on_error(self, mock_settings):
        """Test SQLite transaction rollback on error."""
        import sqlite3

        from src.storage.hybrid_persistence import HybridPersistenceManager

        manager = HybridPersistenceManager(mock_settings)

        # Mock connection methods
        conn = Mock()
        conn.execute = Mock()
        conn.commit = Mock()
        conn.rollback = Mock()
        manager.sqlite_connection = conn

        with pytest.raises(PersistenceError):
            async with manager._sqlite_transaction():
                # Simulate error during transaction
                raise sqlite3.Error("Transaction error")

        # Verify rollback was called
        conn.rollback.assert_called_once()

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_get_document_by_hash_retry_mechanism(self, mock_settings):
        """Test retry mechanism for get_document_by_hash."""
        import sqlite3

        from src.storage.hybrid_persistence import HybridPersistenceManager

        manager = HybridPersistenceManager(mock_settings)

        # Mock cursor to fail first time, succeed second time
        call_count = 0

        def mock_cursor():
            nonlocal call_count
            call_count += 1
            cursor = Mock()
            if call_count == 1:
                cursor.execute.side_effect = sqlite3.OperationalError("Database locked")
            else:
                cursor.fetchone.return_value = None  # No result found
            return cursor

        manager.sqlite_connection = Mock()
        manager.sqlite_connection.cursor = Mock(side_effect=mock_cursor)

        result = await manager.get_document_by_hash("test-hash")

        # Should retry and return result
        assert result is None
        assert call_count == 2  # Should have retried

    @patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", True)
    @patch("src.storage.hybrid_persistence.QdrantClient")
    @pytest.mark.asyncio
    async def test_store_document_retry_mechanism(
        self, mock_qdrant_client, mock_settings, sample_document_metadata
    ):
        """Test retry mechanism for store_document."""
        from src.storage.hybrid_persistence import HybridPersistenceManager

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        manager = HybridPersistenceManager(mock_settings)
        manager.qdrant_client = mock_client

        # Mock SQL connection to fail first time, succeed second time
        call_count = 0
        manager.sqlite_connection = Mock()

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PersistenceError("Transaction failed")
            return None

        manager.sqlite_connection.execute = Mock(side_effect=mock_execute)

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = None

            # Should succeed after retry
            result = await manager.store_document(sample_document_metadata, [])

        assert result is True
        assert call_count >= 2  # Should have retried at least once
