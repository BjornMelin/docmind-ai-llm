"""Test suite for data persistence across restarts and data integrity validation.

This module tests the persistence capabilities of the storage system:
- Data persistence across application restarts
- Database integrity validation
- Recovery from corruption scenarios
- Cross-session data consistency
- WAL mode behavior and recovery
"""

import contextlib
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.models.storage import DocumentMetadata, PersistenceError, VectorRecord
from src.storage.hybrid_persistence import HybridPersistenceManager


@pytest.fixture
def persistent_db_path():
    """Create a persistent database path that survives across test restarts."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    return db_path


@pytest.fixture
def mock_settings_with_path(persistent_db_path):
    """Create mock settings with persistent database path."""
    settings = Mock()
    settings.database.sqlite_db_path = str(persistent_db_path)
    settings.database.qdrant_url = "http://localhost:6333"
    settings.database.qdrant_timeout = 30
    return settings


@pytest.fixture
def sample_documents():
    """Create sample documents for persistence testing."""
    timestamp = time.time()
    return [
        DocumentMetadata(
            id="persist_doc_001",
            file_path="/test/persistence/doc1.pdf",
            file_hash="persist_hash_001",
            file_size=1024000,
            processing_time=2.5,
            strategy_used="hi_res",
            element_count=25,
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"title": "Persistence Test Doc 1", "pages": 5},
        ),
        DocumentMetadata(
            id="persist_doc_002",
            file_path="/test/persistence/doc2.pdf",
            file_hash="persist_hash_002",
            file_size=2048000,
            processing_time=3.7,
            strategy_used="fast",
            element_count=15,
            created_at=timestamp + 10,
            updated_at=timestamp + 10,
            metadata={"title": "Persistence Test Doc 2", "pages": 8},
        ),
    ]


@pytest.fixture
def sample_vectors():
    """Create sample vectors for persistence testing."""
    return [
        VectorRecord(
            id="persist_vec_001",
            document_id="persist_doc_001",
            chunk_index=0,
            text="Persistent vector content for testing restarts.",
            embedding=[0.1, 0.2, 0.3, 0.4] * 256,  # 1024 dimensions
            metadata={"page": 1, "paragraph": 1, "persistent": True},
        ),
        VectorRecord(
            id="persist_vec_002",
            document_id="persist_doc_002",
            chunk_index=0,
            text="Another persistent vector for cross-session testing.",
            embedding=[0.5, 0.6, 0.7, 0.8] * 256,  # 1024 dimensions
            metadata={"page": 2, "paragraph": 3, "persistent": True},
        ),
    ]


class TestDataPersistenceAcrossRestarts:
    """Test suite for data persistence across application restarts."""

    @pytest.mark.integration
    async def test_document_persistence_across_manager_recreation(
        self, mock_settings_with_path, sample_documents, persistent_db_path
    ):
        """Test that documents persist when manager is recreated."""
        try:
            # Phase 1: Create manager and store documents
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager1 = HybridPersistenceManager(mock_settings_with_path)

                for doc in sample_documents:
                    result = await manager1.store_document(doc, [])
                    assert result is True

                await manager1.close()

            # Phase 2: Create new manager instance and verify data persists
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager2 = HybridPersistenceManager(mock_settings_with_path)

                for doc in sample_documents:
                    retrieved = await manager2.get_document_by_hash(doc.file_hash)
                    assert retrieved is not None
                    assert retrieved.id == doc.id
                    assert retrieved.file_path == doc.file_path
                    assert retrieved.metadata == doc.metadata

                await manager2.close()

        finally:
            # Cleanup - remove database files
            if persistent_db_path.exists():
                persistent_db_path.unlink()
            for suffix in ["-wal", "-shm"]:
                wal_path = Path(str(persistent_db_path) + suffix)
                if wal_path.exists():
                    wal_path.unlink()

    @pytest.mark.integration
    async def test_wal_mode_persistence(
        self, mock_settings_with_path, sample_documents, persistent_db_path
    ):
        """Test that WAL mode properly persists data across sessions."""
        try:
            # Phase 1: Store data with WAL mode
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager1 = HybridPersistenceManager(mock_settings_with_path)

                # Verify WAL mode is enabled
                cursor = manager1.sqlite_connection.cursor()
                cursor.execute("PRAGMA journal_mode")
                journal_mode = cursor.fetchone()[0]
                assert journal_mode.lower() == "wal"

                # Store document
                await manager1.store_document(sample_documents[0], [])

                # Verify WAL file is created
                wal_file = Path(str(persistent_db_path) + "-wal")
                assert wal_file.exists()

                await manager1.close()

            # Phase 2: Verify data persists after restart
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager2 = HybridPersistenceManager(mock_settings_with_path)

                retrieved = await manager2.get_document_by_hash(
                    sample_documents[0].file_hash
                )
                assert retrieved is not None
                assert retrieved.id == sample_documents[0].id

                await manager2.close()

        finally:
            # Cleanup
            if persistent_db_path.exists():
                persistent_db_path.unlink()
            for suffix in ["-wal", "-shm"]:
                wal_path = Path(str(persistent_db_path) + suffix)
                if wal_path.exists():
                    wal_path.unlink()

    @pytest.mark.integration
    async def test_concurrent_session_data_visibility(
        self, mock_settings_with_path, sample_documents, persistent_db_path
    ):
        """Test data visibility across concurrent manager instances."""
        try:
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                # Create two managers using the same database
                manager1 = HybridPersistenceManager(mock_settings_with_path)
                manager2 = HybridPersistenceManager(mock_settings_with_path)

                # Store document with first manager
                await manager1.store_document(sample_documents[0], [])

                # Verify second manager can see the data
                retrieved = await manager2.get_document_by_hash(
                    sample_documents[0].file_hash
                )
                assert retrieved is not None
                assert retrieved.id == sample_documents[0].id

                # Store document with second manager
                await manager2.store_document(sample_documents[1], [])

                # Verify first manager can see the new data
                retrieved = await manager1.get_document_by_hash(
                    sample_documents[1].file_hash
                )
                assert retrieved is not None
                assert retrieved.id == sample_documents[1].id

                await manager1.close()
                await manager2.close()

        finally:
            # Cleanup
            if persistent_db_path.exists():
                persistent_db_path.unlink()
            for suffix in ["-wal", "-shm"]:
                wal_path = Path(str(persistent_db_path) + suffix)
                if wal_path.exists():
                    wal_path.unlink()

    @pytest.mark.integration
    async def test_storage_stats_persistence(
        self, mock_settings_with_path, sample_documents, persistent_db_path
    ):
        """Test that storage statistics persist across restarts."""
        try:
            # Phase 1: Store data and get initial stats
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager1 = HybridPersistenceManager(mock_settings_with_path)

                for doc in sample_documents:
                    await manager1.store_document(doc, [])

                stats1 = await manager1.get_storage_stats()
                assert stats1.total_documents == 2
                assert stats1.last_indexed_at is not None

                await manager1.close()

            # Phase 2: Create new manager and verify stats consistency
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager2 = HybridPersistenceManager(mock_settings_with_path)

                stats2 = await manager2.get_storage_stats()
                assert stats2.total_documents == 2
                assert stats2.last_indexed_at == stats1.last_indexed_at

                await manager2.close()

        finally:
            # Cleanup
            if persistent_db_path.exists():
                persistent_db_path.unlink()
            for suffix in ["-wal", "-shm"]:
                wal_path = Path(str(persistent_db_path) + suffix)
                if wal_path.exists():
                    wal_path.unlink()


class TestDatabaseIntegrityValidation:
    """Test suite for database integrity validation."""

    @pytest.mark.integration
    async def test_database_integrity_check(
        self, mock_settings_with_path, sample_documents, persistent_db_path
    ):
        """Test database integrity checking mechanisms."""
        try:
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager = HybridPersistenceManager(mock_settings_with_path)

                # Store documents
                for doc in sample_documents:
                    await manager.store_document(doc, [])

                # Check database integrity
                cursor = manager.sqlite_connection.cursor()
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                assert integrity_result == "ok"

                # Check foreign key constraints
                cursor.execute("PRAGMA foreign_key_check")
                fk_violations = cursor.fetchall()
                assert len(fk_violations) == 0

                await manager.close()

        finally:
            # Cleanup
            if persistent_db_path.exists():
                persistent_db_path.unlink()
            for suffix in ["-wal", "-shm"]:
                wal_path = Path(str(persistent_db_path) + suffix)
                if wal_path.exists():
                    wal_path.unlink()

    @pytest.mark.integration
    async def test_schema_consistency_validation(
        self, mock_settings_with_path, persistent_db_path
    ):
        """Test that database schema remains consistent across restarts."""
        try:
            expected_columns = [
                "id",
                "file_path",
                "file_hash",
                "file_size",
                "processing_time",
                "strategy_used",
                "element_count",
                "created_at",
                "updated_at",
                "metadata",
            ]

            # Phase 1: Create initial schema
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager1 = HybridPersistenceManager(mock_settings_with_path)

                cursor = manager1.sqlite_connection.cursor()
                cursor.execute("PRAGMA table_info(documents)")
                columns1 = [row[1] for row in cursor.fetchall()]

                await manager1.close()

            # Phase 2: Verify schema consistency after restart
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager2 = HybridPersistenceManager(mock_settings_with_path)

                cursor = manager2.sqlite_connection.cursor()
                cursor.execute("PRAGMA table_info(documents)")
                columns2 = [row[1] for row in cursor.fetchall()]

                assert columns1 == columns2
                assert all(col in columns2 for col in expected_columns)

                await manager2.close()

        finally:
            # Cleanup
            if persistent_db_path.exists():
                persistent_db_path.unlink()
            for suffix in ["-wal", "-shm"]:
                wal_path = Path(str(persistent_db_path) + suffix)
                if wal_path.exists():
                    wal_path.unlink()

    @pytest.mark.integration
    async def test_index_consistency_validation(
        self, mock_settings_with_path, persistent_db_path
    ):
        """Test that database indexes remain consistent."""
        try:
            # Phase 1: Create database with indexes
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager1 = HybridPersistenceManager(mock_settings_with_path)

                cursor = manager1.sqlite_connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
                indexes1 = [row[0] for row in cursor.fetchall()]

                await manager1.close()

            # Phase 2: Verify indexes persist
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager2 = HybridPersistenceManager(mock_settings_with_path)

                cursor = manager2.sqlite_connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
                indexes2 = [row[0] for row in cursor.fetchall()]

                assert set(indexes1) == set(indexes2)

                # Verify specific indexes exist
                expected_indexes = ["file_hash", "created_at", "file_path"]
                for expected in expected_indexes:
                    assert any(expected in idx for idx in indexes2)

                await manager2.close()

        finally:
            # Cleanup
            if persistent_db_path.exists():
                persistent_db_path.unlink()
            for suffix in ["-wal", "-shm"]:
                wal_path = Path(str(persistent_db_path) + suffix)
                if wal_path.exists():
                    wal_path.unlink()


class TestCorruptionRecoveryScenarios:
    """Test suite for recovery from database corruption scenarios."""

    @pytest.mark.integration
    async def test_recovery_from_incomplete_transaction(
        self, mock_settings_with_path, sample_documents, persistent_db_path
    ):
        """Test recovery when transaction is incomplete."""
        try:
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager = HybridPersistenceManager(mock_settings_with_path)

                # Simulate incomplete transaction by starting but not committing
                cursor = manager.sqlite_connection.cursor()
                cursor.execute("BEGIN")
                cursor.execute(
                    "INSERT INTO documents (id, file_path, file_hash, file_size, "
                    "processing_time, strategy_used, element_count, created_at, "
                    "updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "incomplete_doc",
                        "/test/incomplete.pdf",
                        "incomplete_hash",
                        1000,
                        1.0,
                        "test",
                        10,
                        time.time(),
                        time.time(),
                        "{}",
                    ),
                )
                # Don't commit - simulate crash

                # Force close without proper cleanup
                manager.sqlite_connection.close()

            # Verify recovery works
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager2 = HybridPersistenceManager(mock_settings_with_path)

                # Should be able to operate normally
                await manager2.store_document(sample_documents[0], [])
                retrieved = await manager2.get_document_by_hash(
                    sample_documents[0].file_hash
                )
                assert retrieved is not None

                # Incomplete transaction should be rolled back
                incomplete = await manager2.get_document_by_hash("incomplete_hash")
                assert incomplete is None

                await manager2.close()

        finally:
            # Cleanup
            if persistent_db_path.exists():
                persistent_db_path.unlink()
            for suffix in ["-wal", "-shm"]:
                wal_path = Path(str(persistent_db_path) + suffix)
                if wal_path.exists():
                    wal_path.unlink()

    @pytest.mark.integration
    async def test_recovery_from_permission_error(
        self, mock_settings_with_path, sample_documents, persistent_db_path
    ):
        """Test recovery when database file permissions cause issues."""
        try:
            # Phase 1: Create database normally
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager1 = HybridPersistenceManager(mock_settings_with_path)
                await manager1.store_document(sample_documents[0], [])
                await manager1.close()

            # Phase 2: Temporarily make database read-only
            persistent_db_path.chmod(0o444)  # Read-only

            # Attempt to create manager (should handle gracefully)
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                with pytest.raises(PersistenceError):
                    HybridPersistenceManager(mock_settings_with_path)

            # Phase 3: Restore permissions and verify recovery
            persistent_db_path.chmod(0o644)  # Read-write

            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager2 = HybridPersistenceManager(mock_settings_with_path)
                retrieved = await manager2.get_document_by_hash(
                    sample_documents[0].file_hash
                )
                assert retrieved is not None
                await manager2.close()

        finally:
            # Ensure permissions are restored for cleanup
            with contextlib.suppress(OSError, PermissionError):
                persistent_db_path.chmod(0o644)

            # Cleanup
            if persistent_db_path.exists():
                persistent_db_path.unlink()
            for suffix in ["-wal", "-shm"]:
                wal_path = Path(str(persistent_db_path) + suffix)
                if wal_path.exists():
                    wal_path.unlink()


class TestCrossSessionDataConsistency:
    """Test suite for cross-session data consistency."""

    @pytest.mark.integration
    async def test_timestamp_consistency_across_sessions(
        self, mock_settings_with_path, sample_documents, persistent_db_path
    ):
        """Test that timestamps remain consistent across sessions."""
        try:
            original_timestamps = {}

            # Phase 1: Store documents and record timestamps
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager1 = HybridPersistenceManager(mock_settings_with_path)

                for doc in sample_documents:
                    await manager1.store_document(doc, [])
                    original_timestamps[doc.id] = {
                        "created_at": doc.created_at,
                        "updated_at": doc.updated_at,
                    }

                await manager1.close()

            # Phase 2: Verify timestamps are preserved
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager2 = HybridPersistenceManager(mock_settings_with_path)

                for doc in sample_documents:
                    retrieved = await manager2.get_document_by_hash(doc.file_hash)
                    assert retrieved is not None

                    original = original_timestamps[doc.id]
                    assert retrieved.created_at == original["created_at"]
                    assert retrieved.updated_at == original["updated_at"]

                await manager2.close()

        finally:
            # Cleanup
            if persistent_db_path.exists():
                persistent_db_path.unlink()
            for suffix in ["-wal", "-shm"]:
                wal_path = Path(str(persistent_db_path) + suffix)
                if wal_path.exists():
                    wal_path.unlink()

    @pytest.mark.integration
    async def test_metadata_serialization_consistency(
        self, mock_settings_with_path, persistent_db_path
    ):
        """Test that complex metadata serialization is consistent."""
        try:
            complex_metadata = {
                "nested": {
                    "array": [1, 2, 3],
                    "string": "test_value",
                    "boolean": True,
                    "null_value": None,
                },
                "unicode": "测试数据 with émojis 🚀",
                "numbers": {"int": 42, "float": 3.14159},
            }

            doc = DocumentMetadata(
                id="complex_metadata_doc",
                file_path="/test/complex.pdf",
                file_hash="complex_hash",
                file_size=1024,
                processing_time=1.0,
                strategy_used="test",
                element_count=10,
                created_at=time.time(),
                updated_at=time.time(),
                metadata=complex_metadata,
            )

            # Phase 1: Store document with complex metadata
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager1 = HybridPersistenceManager(mock_settings_with_path)
                await manager1.store_document(doc, [])
                await manager1.close()

            # Phase 2: Verify metadata deserialization
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager2 = HybridPersistenceManager(mock_settings_with_path)
                retrieved = await manager2.get_document_by_hash(doc.file_hash)

                assert retrieved is not None
                assert retrieved.metadata == complex_metadata
                assert retrieved.metadata["nested"]["array"] == [1, 2, 3]
                assert retrieved.metadata["unicode"] == "测试数据 with émojis 🚀"
                assert retrieved.metadata["numbers"]["float"] == 3.14159

                await manager2.close()

        finally:
            # Cleanup
            if persistent_db_path.exists():
                persistent_db_path.unlink()
            for suffix in ["-wal", "-shm"]:
                wal_path = Path(str(persistent_db_path) + suffix)
                if wal_path.exists():
                    wal_path.unlink()

    @pytest.mark.integration
    async def test_cleanup_operation_persistence(
        self, mock_settings_with_path, sample_documents, persistent_db_path
    ):
        """Test that cleanup operations properly persist."""
        try:
            # Create old and new documents
            old_timestamp = time.time() - (40 * 24 * 3600)  # 40 days ago
            old_doc = DocumentMetadata(
                id="old_persistent_doc",
                file_path="/test/old_persistent.pdf",
                file_hash="old_persistent_hash",
                file_size=1000,
                processing_time=1.0,
                strategy_used="test",
                element_count=10,
                created_at=old_timestamp,
                updated_at=old_timestamp,
            )

            # Phase 1: Store both documents
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager1 = HybridPersistenceManager(mock_settings_with_path)

                await manager1.store_document(old_doc, [])
                await manager1.store_document(sample_documents[0], [])

                # Perform cleanup
                deleted_count = await manager1.cleanup_storage(max_age_days=30)
                assert deleted_count >= 1

                await manager1.close()

            # Phase 2: Verify cleanup persists across restart
            with patch("src.storage.hybrid_persistence.QDRANT_AVAILABLE", False):
                manager2 = HybridPersistenceManager(mock_settings_with_path)

                # Old document should be gone
                old_retrieved = await manager2.get_document_by_hash(
                    "old_persistent_hash"
                )
                assert old_retrieved is None

                # New document should remain
                new_retrieved = await manager2.get_document_by_hash(
                    sample_documents[0].file_hash
                )
                assert new_retrieved is not None

                await manager2.close()

        finally:
            # Cleanup
            if persistent_db_path.exists():
                persistent_db_path.unlink()
            for suffix in ["-wal", "-shm"]:
                wal_path = Path(str(persistent_db_path) + suffix)
                if wal_path.exists():
                    wal_path.unlink()
