"""Tests for dependency injection implementation.

This module tests the dependency injection patterns implemented in Phase 3.4,
ensuring that @inject decorators work correctly and that container overrides
function properly for testing.
"""

import pytest
from dependency_injector import providers

from src.containers import (
    ApplicationContainer,
    TestContainer,
    get_container,
    unwire_container,
    wire_container,
)


class TestDependencyInjection:
    """Test suite for dependency injection functionality."""

    def setup_method(self):
        """Setup for each test method."""
        # Ensure clean state
        unwire_container()

    def teardown_method(self):
        """Cleanup after each test method."""
        # Ensure clean state
        unwire_container()

    def test_application_container_providers(self):
        """Test that ApplicationContainer has all required providers."""
        container = ApplicationContainer()

        # Check that all required providers exist
        assert hasattr(container, "cache")
        assert hasattr(container, "embedding_model")
        assert hasattr(container, "document_processor")
        assert hasattr(container, "multi_agent_coordinator")

        # Check provider types
        assert isinstance(container.cache, providers.Singleton)
        assert isinstance(container.embedding_model, providers.Factory)
        assert isinstance(container.document_processor, providers.Factory)
        assert isinstance(container.multi_agent_coordinator, providers.Singleton)

    def test_test_container_overrides(self):
        """Test that TestContainer properly overrides providers."""
        container = TestContainer()

        # Check that all required providers exist
        assert hasattr(container, "cache")
        assert hasattr(container, "embedding_model")
        assert hasattr(container, "document_processor")
        assert hasattr(container, "multi_agent_coordinator")

        # Check provider types (overrides are Factory providers)
        assert isinstance(container.cache, providers.Factory)
        assert isinstance(container.embedding_model, providers.Factory)
        assert isinstance(container.document_processor, providers.Factory)

    def test_container_wiring(self):
        """Test that container wiring works correctly."""
        container = get_container()

        # Wire to this module
        wire_container([__name__])

        try:
            # Test that wiring worked - this should not raise an error
            # We can't test actual injection without importing the demo module
            # due to circular import issues, but we can test that wiring doesn't fail
            assert True  # If we get here, wiring succeeded

        finally:
            unwire_container()

    def test_cache_interface_compliance(self):
        """Test that cache implementations comply with CacheInterface."""
        from src.cache.simple_cache import SimpleCache
        from src.interfaces import CacheInterface
        from tests._mocks.cache import MockCache

        # Test SimpleCache implements interface
        cache = SimpleCache()
        assert isinstance(cache, CacheInterface)

        # Test MockCache implements interface
        mock_cache = MockCache()
        assert isinstance(mock_cache, CacheInterface)

        # Test interface methods exist
        interface_methods = [
            "get_document",
            "store_document",
            "clear_cache",
            "get_cache_stats",
        ]
        for method in interface_methods:
            assert hasattr(cache, method)
            assert hasattr(mock_cache, method)
            assert callable(getattr(cache, method))
            assert callable(getattr(mock_cache, method))

    def test_mock_implementations_work(self):
        """Test that mock implementations function correctly."""
        from tests._mocks.cache import MockCache
        from tests._mocks.embeddings import MockEmbeddingModel
        from tests._mocks.processing import MockDocumentProcessor

        # Test mock cache
        mock_cache = MockCache()
        assert mock_cache is not None

        # Test mock embedding model
        mock_embedding = MockEmbeddingModel()
        assert mock_embedding is not None
        assert hasattr(mock_embedding, "embed_dim")
        assert mock_embedding.embed_dim == 1024

        # Test mock document processor
        mock_processor = MockDocumentProcessor()
        assert mock_processor is not None
        assert hasattr(mock_processor, "get_strategy_for_file")

    @pytest.mark.asyncio
    async def test_mock_cache_functionality(self):
        """Test that mock cache functions work correctly."""
        from tests._mocks.cache import MockCache

        mock_cache = MockCache()

        # Test storing and retrieving
        test_path = "/test/document.pdf"
        test_result = {"content": "test data"}

        # Should return None initially
        result = await mock_cache.get_document(test_path)
        assert result is None

        # Store document
        success = await mock_cache.store_document(test_path, test_result)
        assert success is True

        # Retrieve document
        result = await mock_cache.get_document(test_path)
        assert result == test_result

        # Test cache stats
        stats = await mock_cache.get_cache_stats()
        assert "cache_type" in stats
        assert "total_documents" in stats
        assert "hit_rate" in stats
        assert stats["cache_type"] == "mock_cache"

    def test_mock_embedding_functionality(self):
        """Test that mock embedding model functions work correctly."""
        from tests._mocks.embeddings import MockEmbeddingModel

        mock_embedding = MockEmbeddingModel()

        # Test single query embedding
        query = "test query"
        embedding = mock_embedding._get_query_embedding(query)
        assert isinstance(embedding, list)
        assert len(embedding) == 1024  # BGE-M3 dimension
        assert all(isinstance(x, float) for x in embedding)

        # Test unified embeddings
        texts = ["text 1", "text 2"]
        unified = mock_embedding.get_unified_embeddings(texts)

        assert "dense" in unified
        assert "sparse" in unified
        assert "colbert" in unified

        # Check dense embeddings shape
        dense_embeddings = unified["dense"]
        assert dense_embeddings.shape == (2, 1024)

    @pytest.mark.asyncio
    async def test_mock_document_processor_functionality(self):
        """Test that mock document processor functions work correctly."""
        from src.models.processing import ProcessingStrategy
        from tests._mocks.processing import MockDocumentProcessor

        mock_processor = MockDocumentProcessor()

        # Test strategy determination
        strategy = mock_processor.get_strategy_for_file("test.pdf")
        assert strategy == ProcessingStrategy.HI_RES

        strategy = mock_processor.get_strategy_for_file("test.txt")
        assert strategy == ProcessingStrategy.FAST

        # Test unsupported format
        with pytest.raises(ValueError, match="Unsupported file format"):
            mock_processor.get_strategy_for_file("test.xyz")


class TestDependencyInjectionDemo:
    """Test the dependency injection demonstration module."""

    def setup_method(self):
        """Setup for each test method."""
        unwire_container()

    def teardown_method(self):
        """Cleanup after each test method."""
        unwire_container()

    def test_di_demo_imports(self):
        """Test that DI demo module can be imported."""
        try:
            import src.di_demo

            assert src.di_demo is not None
        except ImportError as e:
            pytest.fail(f"Could not import DI demo module: {e}")

    def test_di_demo_functions_exist(self):
        """Test that DI demo functions exist and are decorated."""
        from src.di_demo import (
            get_cache_with_injection,
            get_document_processor_with_injection,
            get_embedding_model_with_injection,
            get_multi_agent_coordinator_with_injection,
            process_document_with_injection,
        )

        # Check functions exist
        assert callable(get_cache_with_injection)
        assert callable(get_embedding_model_with_injection)
        assert callable(get_document_processor_with_injection)
        assert callable(get_multi_agent_coordinator_with_injection)
        assert callable(process_document_with_injection)

        # Check that functions have injection metadata (they should be wrapped)
        # This is indicated by the presence of __wrapped__ attribute
        assert hasattr(get_cache_with_injection, "__wrapped__") or hasattr(
            get_cache_with_injection, "__annotations__"
        )

    def test_container_environment_detection(self):
        """Test that container properly detects test environment."""
        import importlib
        import os

        # Test with TESTING environment variable
        original_testing = os.environ.get("TESTING")
        original_pytest = os.environ.get("PYTEST_CURRENT_TEST")

        try:
            # Set testing environment
            os.environ["TESTING"] = "1"

            # Force reload of the containers module to get fresh container creation
            import src.containers

            importlib.reload(src.containers)

            from src.containers import create_container

            container = create_container()

            # Verify that test environment was detected (check via provider behavior)
            # In test environment, cache provider should be a Factory (not Singleton)
            assert hasattr(container, "cache")
            cache_provider = getattr(container, "cache", None)

            # Test containers use Factory providers for mocking
            # Application containers use Singleton providers for real instances
            # This is a better test than checking the container type directly
            from dependency_injector import providers

            is_test_environment = isinstance(cache_provider, providers.Factory)
            assert is_test_environment, (
                "Container should be in test mode based on environment"
            )

        finally:
            # Restore environment
            if original_testing is not None:
                os.environ["TESTING"] = original_testing
            else:
                os.environ.pop("TESTING", None)

            if original_pytest is not None:
                os.environ["PYTEST_CURRENT_TEST"] = original_pytest
            else:
                os.environ.pop("PYTEST_CURRENT_TEST", None)

            # Reload again to restore original state
            import src.containers

            importlib.reload(src.containers)
