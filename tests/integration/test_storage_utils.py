"""Comprehensive test suite for storage utility functions and context managers.

This module tests the storage utilities in src/utils/storage.py focusing on:
- Qdrant client creation and management
- Database operations and hybrid collection setup
- Resource management context managers
- GPU memory management utilities
- CUDA error handling and safe operations
- Model lifecycle management
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.utils.storage import (
    async_gpu_memory_context,
    clear_collection,
    create_async_client,
    create_sync_client,
    create_vector_store,
    cuda_error_context,
    get_collection_info,
    get_safe_gpu_info,
    get_safe_vram_usage,
    gpu_memory_context,
    model_context,
    safe_cuda_operation,
    setup_hybrid_collection,
    setup_hybrid_collection_async,
    sync_model_context,
    test_connection,
)


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock()
    settings.database.qdrant_url = "http://localhost:6333"
    settings.database.qdrant_timeout = 30
    settings.embedding.dimension = 1024
    settings.monitoring.default_batch_size = 100
    settings.monitoring.bytes_to_gb_divisor = 1024**3
    return settings


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client for testing."""
    client = Mock()
    client.close = Mock()
    client.collection_exists = Mock(return_value=False)
    client.create_collection = Mock()
    client.delete_collection = Mock()
    client.get_collection = Mock()
    client.get_collections = Mock()
    return client


class TestQdrantClientManagement:
    """Test suite for Qdrant client creation and management."""

    @pytest.mark.integration
    @patch("src.utils.storage.QdrantClient")
    def test_create_sync_client_success(self, mock_client_class, mock_settings):
        """Test successful sync client creation."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        with patch("src.utils.storage.settings", mock_settings):
            with create_sync_client() as client:
                assert client == mock_client
                mock_client_class.assert_called_once_with(
                    url=mock_settings.database.qdrant_url,
                    timeout=mock_settings.database.qdrant_timeout,
                    prefer_grpc=True,
                )

            # Verify cleanup was called
            mock_client.close.assert_called_once()

    @pytest.mark.integration
    @patch("src.utils.storage.QdrantClient")
    def test_create_sync_client_connection_error(self, mock_client_class):
        """Test sync client creation with connection error."""
        mock_client_class.side_effect = ConnectionError("Cannot connect to Qdrant")

        with pytest.raises(ConnectionError):
            with create_sync_client():
                pass

    @pytest.mark.integration
    @patch("src.utils.storage.QdrantClient")
    def test_create_sync_client_cleanup_on_error(
        self, mock_client_class, mock_settings
    ):
        """Test sync client cleanup when error occurs during context."""
        mock_client = Mock()
        mock_client.close.side_effect = ConnectionError("Close failed")
        mock_client_class.return_value = mock_client

        with patch("src.utils.storage.settings", mock_settings):
            try:
                with create_sync_client() as client:
                    raise ValueError("Test error")
            except ValueError:
                pass

            # Verify close was attempted despite error
            mock_client.close.assert_called_once()

    @pytest.mark.integration
    @patch("src.utils.storage.AsyncQdrantClient")
    async def test_create_async_client_success(self, mock_client_class, mock_settings):
        """Test successful async client creation."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        with patch("src.utils.storage.settings", mock_settings):
            async with create_async_client() as client:
                assert client == mock_client
                mock_client_class.assert_called_once_with(
                    url=mock_settings.database.qdrant_url,
                    timeout=mock_settings.database.qdrant_timeout,
                    prefer_grpc=True,
                )

            # Verify cleanup was called
            mock_client.close.assert_called_once()

    @pytest.mark.integration
    @patch("src.utils.storage.AsyncQdrantClient")
    async def test_create_async_client_connection_error(self, mock_client_class):
        """Test async client creation with connection error."""
        mock_client_class.side_effect = TimeoutError("Connection timeout")

        with pytest.raises(TimeoutError):
            async with create_async_client():
                pass


class TestHybridCollectionSetup:
    """Test suite for hybrid collection setup operations."""

    @pytest.mark.integration
    @patch("src.utils.storage.QdrantVectorStore")
    async def test_setup_hybrid_collection_async_new_collection(
        self, mock_vector_store_class, mock_settings
    ):
        """Test async hybrid collection setup for new collection."""
        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = False
        mock_client.create_collection = AsyncMock()

        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        with patch("src.utils.storage.settings", mock_settings):
            with patch("src.utils.storage.QdrantClient") as mock_sync_client:
                result = await setup_hybrid_collection_async(
                    mock_client, "test_collection", dense_embedding_size=1024
                )

                assert result == mock_vector_store
                mock_client.collection_exists.assert_called_once_with("test_collection")
                mock_client.create_collection.assert_called_once()

    @pytest.mark.integration
    @patch("src.utils.storage.QdrantVectorStore")
    async def test_setup_hybrid_collection_async_existing_collection(
        self, mock_vector_store_class, mock_settings
    ):
        """Test async hybrid collection setup for existing collection."""
        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = True

        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        with patch("src.utils.storage.settings", mock_settings):
            with patch("src.utils.storage.QdrantClient"):
                result = await setup_hybrid_collection_async(
                    mock_client, "existing_collection"
                )

                assert result == mock_vector_store
                mock_client.collection_exists.assert_called_once_with(
                    "existing_collection"
                )
                mock_client.create_collection.assert_not_called()

    @pytest.mark.integration
    @patch("src.utils.storage.QdrantVectorStore")
    async def test_setup_hybrid_collection_async_recreate(
        self, mock_vector_store_class, mock_settings
    ):
        """Test async hybrid collection setup with recreation."""
        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = True
        mock_client.delete_collection = AsyncMock()
        mock_client.create_collection = AsyncMock()

        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        with patch("src.utils.storage.settings", mock_settings):
            with patch("src.utils.storage.QdrantClient"):
                result = await setup_hybrid_collection_async(
                    mock_client, "recreate_collection", recreate=True
                )

                assert result == mock_vector_store
                mock_client.delete_collection.assert_called_once_with(
                    "recreate_collection"
                )
                mock_client.create_collection.assert_called_once()

    @pytest.mark.integration
    @patch("src.utils.storage.QdrantVectorStore")
    def test_setup_hybrid_collection_sync(self, mock_vector_store_class, mock_settings):
        """Test sync hybrid collection setup."""
        mock_client = Mock()
        mock_client.collection_exists.return_value = False

        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        with patch("src.utils.storage.settings", mock_settings):
            result = setup_hybrid_collection(mock_client, "sync_collection")

            assert result == mock_vector_store
            mock_client.collection_exists.assert_called_once_with("sync_collection")
            mock_client.create_collection.assert_called_once()


class TestDatabaseOperations:
    """Test suite for database utility operations."""

    @pytest.mark.integration
    @patch("src.utils.storage.create_sync_client")
    def test_get_collection_info_success(self, mock_create_client):
        """Test successful collection info retrieval."""
        mock_client = Mock()
        mock_collection_info = Mock()
        mock_collection_info.points_count = 1000
        mock_collection_info.config.params.vectors = {"dense": "config"}
        mock_collection_info.config.params.sparse_vectors = {"sparse": "config"}
        mock_collection_info.status = "green"

        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = mock_collection_info

        mock_context_manager = Mock()
        mock_context_manager.__enter__.return_value = mock_client
        mock_context_manager.__exit__.return_value = None
        mock_create_client.return_value = mock_context_manager

        info = get_collection_info("test_collection")

        assert info["exists"] is True
        assert info["points_count"] == 1000
        assert "vectors_config" in info
        assert "sparse_vectors_config" in info
        assert info["status"] == "green"

    @pytest.mark.integration
    @patch("src.utils.storage.create_sync_client")
    def test_get_collection_info_not_exists(self, mock_create_client):
        """Test collection info retrieval for non-existent collection."""
        mock_client = Mock()
        mock_client.collection_exists.return_value = False

        mock_context_manager = Mock()
        mock_context_manager.__enter__.return_value = mock_client
        mock_context_manager.__exit__.return_value = None
        mock_create_client.return_value = mock_context_manager

        info = get_collection_info("nonexistent_collection")

        assert info["exists"] is False
        assert info["error"] == "Collection not found"

    @pytest.mark.integration
    @patch("src.utils.storage.create_sync_client")
    def test_test_connection_success(self, mock_create_client, mock_settings):
        """Test successful database connection test."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = [
            Mock(name="collection1"),
            Mock(name="collection2"),
        ]
        mock_client.get_collections.return_value = mock_collections

        mock_context_manager = Mock()
        mock_context_manager.__enter__.return_value = mock_client
        mock_context_manager.__exit__.return_value = None
        mock_create_client.return_value = mock_context_manager

        with patch("src.utils.storage.settings", mock_settings):
            result = test_connection()

            assert result["connected"] is True
            assert result["url"] == mock_settings.database.qdrant_url
            assert result["collections_count"] == 2
            assert "collection1" in result["collections"]
            assert "collection2" in result["collections"]

    @pytest.mark.integration
    @patch("src.utils.storage.create_sync_client")
    def test_test_connection_failure(self, mock_create_client, mock_settings):
        """Test database connection test failure."""
        mock_create_client.side_effect = ConnectionError("Connection failed")

        with patch("src.utils.storage.settings", mock_settings):
            result = test_connection()

            assert result["connected"] is False
            assert result["url"] == mock_settings.database.qdrant_url
            assert "error" in result

    @pytest.mark.integration
    @patch("src.utils.storage.create_sync_client")
    def test_clear_collection_success(self, mock_create_client):
        """Test successful collection clearing."""
        mock_client = Mock()
        mock_client.collection_exists.return_value = True
        mock_collection_info = Mock()
        mock_collection_info.config.params.vectors = {"test": "config"}
        mock_collection_info.config.params.sparse_vectors = {"sparse": "config"}
        mock_client.get_collection.return_value = mock_collection_info

        mock_context_manager = Mock()
        mock_context_manager.__enter__.return_value = mock_client
        mock_context_manager.__exit__.return_value = None
        mock_create_client.return_value = mock_context_manager

        result = clear_collection("test_collection")

        assert result is True
        mock_client.delete_collection.assert_called_once_with("test_collection")
        mock_client.create_collection.assert_called_once()

    @pytest.mark.integration
    @patch("src.utils.storage.create_sync_client")
    def test_clear_collection_not_exists(self, mock_create_client):
        """Test clearing non-existent collection."""
        mock_client = Mock()
        mock_client.collection_exists.return_value = False

        mock_context_manager = Mock()
        mock_context_manager.__enter__.return_value = mock_client
        mock_context_manager.__exit__.return_value = None
        mock_create_client.return_value = mock_context_manager

        result = clear_collection("nonexistent_collection")

        assert result is False


class TestGPUMemoryManagement:
    """Test suite for GPU memory management utilities."""

    @pytest.mark.integration
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.empty_cache")
    @patch("gc.collect")
    def test_gpu_memory_context_success(
        self, mock_gc_collect, mock_empty_cache, mock_synchronize, mock_cuda_available
    ):
        """Test GPU memory context manager success case."""
        mock_cuda_available.return_value = True

        with gpu_memory_context():
            # Simulate GPU operation
            pass

        mock_synchronize.assert_called_once()
        mock_empty_cache.assert_called_once()
        mock_gc_collect.assert_called_once()

    @pytest.mark.integration
    @patch("torch.cuda.is_available")
    @patch("gc.collect")
    def test_gpu_memory_context_no_cuda(self, mock_gc_collect, mock_cuda_available):
        """Test GPU memory context when CUDA is not available."""
        mock_cuda_available.return_value = False

        with gpu_memory_context():
            pass

        mock_gc_collect.assert_called_once()

    @pytest.mark.integration
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.synchronize")
    @patch("gc.collect")
    def test_gpu_memory_context_with_error(
        self, mock_gc_collect, mock_synchronize, mock_cuda_available
    ):
        """Test GPU memory context handles errors gracefully."""
        mock_cuda_available.return_value = True
        mock_synchronize.side_effect = RuntimeError("CUDA error")

        with gpu_memory_context():
            pass

        # Should still call garbage collection despite error
        mock_gc_collect.assert_called_once()

    @pytest.mark.integration
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.empty_cache")
    @patch("gc.collect")
    async def test_async_gpu_memory_context(
        self, mock_gc_collect, mock_empty_cache, mock_synchronize, mock_cuda_available
    ):
        """Test async GPU memory context manager."""
        mock_cuda_available.return_value = True

        async with async_gpu_memory_context():
            # Simulate async GPU operation
            await asyncio.sleep(0.001)

        mock_synchronize.assert_called_once()
        mock_empty_cache.assert_called_once()
        mock_gc_collect.assert_called_once()

    @pytest.mark.integration
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.synchronize")
    @patch("gc.collect")
    async def test_async_gpu_memory_context_error_handling(
        self, mock_gc_collect, mock_synchronize, mock_cuda_available
    ):
        """Test async GPU memory context error handling."""
        mock_cuda_available.return_value = True
        mock_synchronize.side_effect = RuntimeError("CUDA error")

        async with async_gpu_memory_context():
            pass

        mock_gc_collect.assert_called_once()


class TestModelContextManagement:
    """Test suite for model context management."""

    @pytest.mark.integration
    async def test_model_context_sync_factory(self):
        """Test model context with sync factory function."""

        def create_test_model():
            model = Mock()
            model.cleanup = Mock()
            return model

        async with model_context(create_test_model, cleanup_method="cleanup") as model:
            assert model is not None
            assert hasattr(model, "cleanup")

        # Verify cleanup was called
        model.cleanup.assert_called_once()

    @pytest.mark.integration
    async def test_model_context_async_factory(self):
        """Test model context with async factory function."""

        async def create_async_model():
            model = Mock()
            model.cleanup = AsyncMock()
            return model

        async with model_context(create_async_model, cleanup_method="cleanup") as model:
            assert model is not None
            assert hasattr(model, "cleanup")

        # Verify async cleanup was called
        model.cleanup.assert_called_once()

    @pytest.mark.integration
    async def test_model_context_no_cleanup_method(self):
        """Test model context when no cleanup method specified."""

        def create_simple_model():
            return Mock()

        async with model_context(create_simple_model) as model:
            assert model is not None

        # Should complete without error even without cleanup

    @pytest.mark.integration
    async def test_model_context_cleanup_error(self):
        """Test model context handles cleanup errors gracefully."""

        def create_error_model():
            model = Mock()
            model.cleanup = Mock(side_effect=Exception("Cleanup failed"))
            return model

        async with model_context(create_error_model, cleanup_method="cleanup") as model:
            assert model is not None

        # Should complete despite cleanup error
        model.cleanup.assert_called_once()

    @pytest.mark.integration
    def test_sync_model_context(self):
        """Test synchronous model context manager."""

        def create_sync_model():
            model = Mock()
            model.close = Mock()
            return model

        with sync_model_context(create_sync_model, cleanup_method="close") as model:
            assert model is not None
            assert hasattr(model, "close")

        model.close.assert_called_once()

    @pytest.mark.integration
    def test_sync_model_context_cleanup_error(self):
        """Test sync model context handles cleanup errors."""

        def create_error_model():
            model = Mock()
            model.close = Mock(side_effect=AttributeError("No close method"))
            return model

        with sync_model_context(create_error_model, cleanup_method="close") as model:
            assert model is not None

        model.close.assert_called_once()


class TestCUDAErrorHandling:
    """Test suite for CUDA error handling utilities."""

    @pytest.mark.integration
    @patch("torch.cuda.is_available")
    def test_cuda_error_context_success(self, mock_cuda_available):
        """Test CUDA error context manager success case."""
        mock_cuda_available.return_value = True

        with cuda_error_context("Test operation") as ctx:
            ctx["result"] = 42

        assert ctx["result"] == 42

    @pytest.mark.integration
    @patch("torch.cuda.is_available")
    def test_cuda_error_context_cuda_error(self, mock_cuda_available):
        """Test CUDA error context handles CUDA errors."""
        mock_cuda_available.return_value = True

        with cuda_error_context(
            "CUDA operation", reraise=False, default_return=0
        ) as ctx:
            raise RuntimeError("CUDA out of memory")

        assert ctx["result"] == 0
        assert "error" in ctx

    @pytest.mark.integration
    def test_cuda_error_context_runtime_error(self):
        """Test CUDA error context handles general runtime errors."""
        with cuda_error_context(
            "Runtime operation", reraise=False, default_return=-1
        ) as ctx:
            raise RuntimeError("General runtime error")

        assert ctx["result"] == -1
        assert "error" in ctx

    @pytest.mark.integration
    def test_cuda_error_context_reraise(self):
        """Test CUDA error context reraises when configured."""
        with pytest.raises(RuntimeError):
            with cuda_error_context("Reraise operation", reraise=True):
                raise RuntimeError("Should be reraised")

    @pytest.mark.integration
    @patch("torch.cuda.is_available")
    def test_safe_cuda_operation_success(self, mock_cuda_available):
        """Test safe CUDA operation success."""
        mock_cuda_available.return_value = True

        def test_operation():
            return "success"

        result = safe_cuda_operation(test_operation, "Test op")
        assert result == "success"

    @pytest.mark.integration
    def test_safe_cuda_operation_error(self):
        """Test safe CUDA operation with error."""

        def failing_operation():
            raise RuntimeError("CUDA error")

        result = safe_cuda_operation(
            failing_operation, "Failing op", default_return="default"
        )
        assert result == "default"

    @pytest.mark.integration
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    def test_get_safe_vram_usage_success(
        self, mock_memory_allocated, mock_cuda_available
    ):
        """Test safe VRAM usage retrieval success."""
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 1024**3  # 1GB

        with patch("src.utils.storage.settings") as mock_settings:
            mock_settings.monitoring.bytes_to_gb_divisor = 1024**3
            vram = get_safe_vram_usage()
            assert vram == 1.0

    @pytest.mark.integration
    @patch("torch.cuda.is_available")
    def test_get_safe_vram_usage_no_cuda(self, mock_cuda_available):
        """Test safe VRAM usage when CUDA not available."""
        mock_cuda_available.return_value = False
        vram = get_safe_vram_usage()
        assert vram == 0.0

    @pytest.mark.integration
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.device_count")
    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated")
    def test_get_safe_gpu_info_success(
        self,
        mock_memory_allocated,
        mock_device_properties,
        mock_device_name,
        mock_device_count,
        mock_cuda_available,
    ):
        """Test safe GPU info retrieval success."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        mock_device_name.return_value = "GeForce RTX 4090"

        mock_props = Mock()
        mock_props.major = 8
        mock_props.minor = 9
        mock_props.total_memory = 24 * 1024**3  # 24GB
        mock_device_properties.return_value = mock_props

        mock_memory_allocated.return_value = 12 * 1024**3  # 12GB

        with patch("src.utils.storage.settings") as mock_settings:
            mock_settings.monitoring.bytes_to_gb_divisor = 1024**3

            info = get_safe_gpu_info()

            assert info["cuda_available"] is True
            assert info["device_count"] == 1
            assert info["device_name"] == "GeForce RTX 4090"
            assert info["compute_capability"] == "8.9"
            assert info["total_memory_gb"] == 24.0
            assert info["allocated_memory_gb"] == 12.0

    @pytest.mark.integration
    @patch("torch.cuda.is_available")
    def test_get_safe_gpu_info_no_cuda(self, mock_cuda_available):
        """Test safe GPU info when CUDA not available."""
        mock_cuda_available.return_value = False

        info = get_safe_gpu_info()

        assert info["cuda_available"] is False
        assert info["device_count"] == 0
        assert info["device_name"] == "Unknown"
        assert info["total_memory_gb"] == 0.0
        assert info["allocated_memory_gb"] == 0.0


class TestVectorStoreCreation:
    """Test suite for vector store creation utilities."""

    @pytest.mark.integration
    @patch("src.utils.storage.QdrantClient")
    @patch("src.utils.storage.QdrantVectorStore")
    def test_create_vector_store_success(
        self, mock_vector_store_class, mock_client_class, mock_settings
    ):
        """Test successful vector store creation."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        with patch("src.utils.storage.settings", mock_settings):
            result = create_vector_store("test_collection", enable_hybrid=True)

            assert result == mock_vector_store
            mock_client_class.assert_called_once_with(
                url=mock_settings.database.qdrant_url
            )
            mock_vector_store_class.assert_called_once_with(
                client=mock_client,
                collection_name="test_collection",
                enable_hybrid=True,
                batch_size=mock_settings.monitoring.default_batch_size,
            )

    @pytest.mark.integration
    @patch("src.utils.storage.QdrantClient")
    @patch("src.utils.storage.QdrantVectorStore")
    def test_create_vector_store_hybrid_disabled(
        self, mock_vector_store_class, mock_client_class, mock_settings
    ):
        """Test vector store creation with hybrid disabled."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        with patch("src.utils.storage.settings", mock_settings):
            result = create_vector_store("simple_collection", enable_hybrid=False)

            assert result == mock_vector_store
            mock_vector_store_class.assert_called_once_with(
                client=mock_client,
                collection_name="simple_collection",
                enable_hybrid=False,
                batch_size=mock_settings.monitoring.default_batch_size,
            )
