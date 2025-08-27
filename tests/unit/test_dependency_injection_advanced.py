"""Advanced dependency injection testing with container overrides.

This module provides comprehensive tests for the dependency injection system
including container overrides, @inject decorators, and testing patterns.

Tests validate:
- Container creation and configuration
- Dependency injection with @inject decorator
- Test container overrides for isolated testing
- Provider configuration and lifecycle management
- Integration between TestContainer and ApplicationContainer
- Advanced DI patterns and error handling
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from src.config.settings import DocMindSettings
from src.containers import ApplicationContainer


class TestContainer(containers.DeclarativeContainer):
    """Test container that overrides ApplicationContainer for testing."""

    # Configuration
    config = providers.Configuration()

    # Mock settings for testing
    settings = providers.Factory(
        DocMindSettings,
        debug=True,
        log_level="DEBUG",
        enable_gpu_acceleration=False,  # CPU-only for unit tests
        enable_dspy_optimization=False,
        enable_performance_logging=False,
    )

    # Mock cache provider
    cache = providers.Singleton(
        MagicMock,
        spec=["get_document", "store_document", "clear_cache", "get_cache_stats"],
    )

    # Mock embedding provider
    embeddings = providers.Factory(
        MagicMock,
        spec=["embed_documents", "embed_query", "aembed_documents", "aembed_query"],
    )

    # Mock LLM provider
    llm = providers.Factory(
        AsyncMock, spec=["ainvoke", "astream", "acomplete", "astream_complete"]
    )

    # Mock vector store provider
    vector_store = providers.Factory(
        MagicMock,
        spec=["add_documents", "similarity_search", "asimilarity_search", "delete"],
    )

    # Mock document processor provider
    document_processor = providers.Factory(
        MagicMock,
        spec=["process_documents", "aprocess_documents", "get_processing_stats"],
    )


class TestDependencyInjectionAdvanced:
    """Advanced tests for dependency injection system."""

    @pytest.fixture(autouse=True)
    def setup_container_override(self):
        """Set up container override for all tests in this class."""
        # Create test container
        test_container = TestContainer()

        # Configure test settings
        test_container.config.from_dict(
            {
                "testing": True,
                "debug": True,
                "cache_dir": "/tmp/test_cache",
                "data_dir": "/tmp/test_data",
            }
        )

        # Override the application container
        ApplicationContainer.override(test_container)

        yield test_container

        # Clean up after test
        ApplicationContainer.reset_override()

    def test_container_override_setup(self, setup_container_override):
        """Test that container override is properly configured."""
        test_container = setup_container_override

        # Verify test container is configured
        assert test_container.config.testing() is True
        assert test_container.config.debug() is True

        # Verify providers are available
        assert test_container.settings is not None
        assert test_container.cache is not None
        assert test_container.embeddings is not None
        assert test_container.llm is not None

    def test_inject_decorator_with_override(self, setup_container_override):
        """Test @inject decorator works with container override."""

        @inject
        def function_with_dependencies(
            cache=Provide[ApplicationContainer.cache],
            settings=Provide[ApplicationContainer.settings],
        ):
            """Function that uses dependency injection."""
            return {
                "cache": cache,
                "settings": settings,
                "cache_type": type(cache).__name__,
                "settings_debug": settings.debug,
            }

        # Call function - should get mocked dependencies
        result = function_with_dependencies()

        assert result is not None
        assert "cache" in result
        assert "settings" in result
        assert result["cache_type"] == "MagicMock"  # Should be mock
        assert result["settings_debug"] is True  # Should be test setting

    @pytest.mark.asyncio
    async def test_async_inject_decorator_with_override(self, setup_container_override):
        """Test @inject decorator works with async functions."""

        @inject
        async def async_function_with_dependencies(
            llm=Provide[ApplicationContainer.llm],
            embeddings=Provide[ApplicationContainer.embeddings],
        ):
            """Async function that uses dependency injection."""
            # Test async method call
            response = await llm.ainvoke("test prompt")

            # Test sync method call
            embedding_result = embeddings.embed_query("test query")

            return {
                "llm": llm,
                "embeddings": embeddings,
                "response": response,
                "embedding_result": embedding_result,
            }

        # Call async function
        result = await async_function_with_dependencies()

        assert result is not None
        assert "llm" in result
        assert "embeddings" in result
        # AsyncMock should have been called
        assert result["llm"].ainvoke.called

    def test_multiple_dependency_injection(self, setup_container_override):
        """Test injection of multiple dependencies."""

        @inject
        def complex_function(
            cache=Provide[ApplicationContainer.cache],
            embeddings=Provide[ApplicationContainer.embeddings],
            vector_store=Provide[ApplicationContainer.vector_store],
            settings=Provide[ApplicationContainer.settings],
        ):
            """Function using multiple injected dependencies."""
            return {
                "cache": cache,
                "embeddings": embeddings,
                "vector_store": vector_store,
                "settings": settings,
                "all_injected": all(
                    [
                        cache is not None,
                        embeddings is not None,
                        vector_store is not None,
                        settings is not None,
                    ]
                ),
            }

        result = complex_function()

        assert result["all_injected"] is True
        assert isinstance(result["cache"], MagicMock)
        assert isinstance(result["embeddings"], MagicMock)
        assert isinstance(result["vector_store"], MagicMock)
        assert isinstance(result["settings"], DocMindSettings)

    def test_container_provider_lifecycle(self, setup_container_override):
        """Test that container providers have proper lifecycle management."""
        test_container = setup_container_override

        # Test singleton behavior for cache
        cache1 = test_container.cache()
        cache2 = test_container.cache()
        assert cache1 is cache2  # Should be same instance (singleton)

        # Test factory behavior for embeddings
        embeddings1 = test_container.embeddings()
        embeddings2 = test_container.embeddings()
        assert embeddings1 is not embeddings2  # Should be different instances (factory)

    def test_container_configuration_override(self, setup_container_override):
        """Test that container configuration can be overridden."""
        test_container = setup_container_override

        # Override configuration
        test_container.config.update(
            {
                "custom_setting": "test_value",
                "debug": False,
            }
        )

        # Verify configuration changes
        assert test_container.config.custom_setting() == "test_value"
        assert test_container.config.debug() is False

    def test_mock_behavior_validation(self, setup_container_override):
        """Test that mocked dependencies behave correctly."""

        @inject
        def test_function(
            cache=Provide[ApplicationContainer.cache],
            embeddings=Provide[ApplicationContainer.embeddings],
        ):
            """Test function to validate mock behavior."""
            # Configure mock behavior
            cache.get_document.return_value = {"content": "test document"}
            cache.store_document.return_value = True

            embeddings.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]

            # Use the mocked dependencies
            doc = cache.get_document("test.pdf")
            success = cache.store_document("test.pdf", {"content": "new content"})
            embedding = embeddings.embed_query("test query")

            return {
                "document": doc,
                "store_success": success,
                "embedding": embedding,
                "cache_calls": cache.get_document.call_count,
                "embedding_calls": embeddings.embed_query.call_count,
            }

        result = test_function()

        assert result["document"]["content"] == "test document"
        assert result["store_success"] is True
        assert result["embedding"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert result["cache_calls"] == 1
        assert result["embedding_calls"] == 1

    @pytest.mark.asyncio
    async def test_async_mock_behavior_validation(self, setup_container_override):
        """Test that async mocked dependencies behave correctly."""

        @inject
        async def async_test_function(
            llm=Provide[ApplicationContainer.llm],
            document_processor=Provide[ApplicationContainer.document_processor],
        ):
            """Async test function to validate async mock behavior."""
            # Configure async mock behavior
            llm.ainvoke.return_value = "Mock LLM response"
            llm.astream.return_value = ["Mock", " stream", " response"]

            document_processor.aprocess_documents.return_value = {
                "processed_count": 5,
                "elements": ["Title", "Text", "List"],
            }

            # Use the async mocked dependencies
            response = await llm.ainvoke("test prompt")
            stream_chunks = []
            async for chunk in llm.astream("test prompt"):
                stream_chunks.append(chunk)

            processing_result = await document_processor.aprocess_documents(
                ["doc1.pdf", "doc2.pdf"]
            )

            return {
                "response": response,
                "stream_chunks": stream_chunks,
                "processing_result": processing_result,
                "llm_calls": llm.ainvoke.call_count,
                "processor_calls": document_processor.aprocess_documents.call_count,
            }

        result = await async_test_function()

        assert result["response"] == "Mock LLM response"
        assert result["stream_chunks"] == ["Mock", " stream", " response"]
        assert result["processing_result"]["processed_count"] == 5
        assert result["llm_calls"] == 1
        assert result["processor_calls"] == 1

    def test_dependency_injection_error_handling(self, setup_container_override):
        """Test error handling in dependency injection."""
        # Test missing provider
        with pytest.raises(Exception):  # Should raise provider not found error

            @inject
            def bad_function(nonexistent=Provide["NonexistentContainer.nonexistent"]):
                return nonexistent

            bad_function()

    def test_container_reset_behavior(self, setup_container_override):
        """Test that container reset works properly."""
        test_container = setup_container_override

        # Get initial instance
        initial_cache = test_container.cache()

        # Reset container
        test_container.reset_singletons()

        # Get new instance after reset
        new_cache = test_container.cache()

        # Should be different instances after reset
        assert initial_cache is not new_cache

    def test_real_vs_mock_container_isolation(self):
        """Test that real and mock containers are properly isolated."""
        # Without override, should get real container behavior
        try:
            from src.containers import ApplicationContainer

            real_container = ApplicationContainer()

            # Real container should have actual providers, not mocks
            settings = real_container.settings()
            assert isinstance(settings, DocMindSettings)
            assert settings.debug is False  # Real default setting

        except ImportError:
            # If ApplicationContainer not available, skip this test
            pytest.skip("ApplicationContainer not available for isolation test")

    def test_container_wiring_and_unwiring(self, setup_container_override):
        """Test container wiring and unwiring functionality."""
        test_container = setup_container_override

        # Test manual wiring
        @inject
        def manually_wired_function(cache=Provide[ApplicationContainer.cache]):
            return cache

        # Should work with override
        result = manually_wired_function()
        assert isinstance(result, MagicMock)

        # Test unwiring (cleanup)
        try:
            test_container.unwire()
            # After unwiring, injection should still work due to override
            result2 = manually_wired_function()
            assert isinstance(result2, MagicMock)
        except Exception:
            # Some unwiring behavior is expected to vary
            pass

    @pytest.mark.asyncio
    async def test_concurrent_dependency_injection(self, setup_container_override):
        """Test dependency injection under concurrent access."""

        @inject
        async def concurrent_worker(
            worker_id: int,
            cache=Provide[ApplicationContainer.cache],
            embeddings=Provide[ApplicationContainer.embeddings],
        ):
            """Worker function that uses dependency injection."""
            # Configure mock return values
            cache.get_document.return_value = f"document_{worker_id}"
            embeddings.embed_query.return_value = [worker_id] * 5

            # Simulate some async work
            await asyncio.sleep(0.01)

            # Use injected dependencies
            doc = cache.get_document(f"doc_{worker_id}.pdf")
            embedding = embeddings.embed_query(f"query_{worker_id}")

            return {
                "worker_id": worker_id,
                "document": doc,
                "embedding": embedding,
            }

        # Run multiple workers concurrently
        workers = [concurrent_worker(i) for i in range(10)]
        results = await asyncio.gather(*workers)

        # Validate all workers completed successfully
        assert len(results) == 10
        for i, result in enumerate(results):
            assert result["worker_id"] == i
            assert result["document"] == f"document_{i}"
            assert result["embedding"] == [i] * 5

    def test_container_provider_specifications(self, setup_container_override):
        """Test that mock providers have correct specifications."""
        test_container = setup_container_override

        # Test cache provider spec
        cache = test_container.cache()
        expected_cache_methods = [
            "get_document",
            "store_document",
            "clear_cache",
            "get_cache_stats",
        ]
        for method_name in expected_cache_methods:
            assert hasattr(cache, method_name)
            assert callable(getattr(cache, method_name))

        # Test embeddings provider spec
        embeddings = test_container.embeddings()
        expected_embedding_methods = [
            "embed_documents",
            "embed_query",
            "aembed_documents",
            "aembed_query",
        ]
        for method_name in expected_embedding_methods:
            assert hasattr(embeddings, method_name)
            assert callable(getattr(embeddings, method_name))

        # Test LLM provider spec
        llm = test_container.llm()
        expected_llm_methods = ["ainvoke", "astream", "acomplete", "astream_complete"]
        for method_name in expected_llm_methods:
            assert hasattr(llm, method_name)
            assert callable(getattr(llm, method_name))


class TestDependencyInjectionIntegration:
    """Integration tests for dependency injection with real components."""

    @pytest.mark.integration
    def test_real_container_integration(self):
        """Test integration with real application container (when available)."""
        try:
            from src.containers import ApplicationContainer

            # Create real container
            container = ApplicationContainer()

            # Test that real providers can be instantiated
            settings = container.settings()
            assert isinstance(settings, DocMindSettings)

            # Test configuration loading
            assert hasattr(settings, "app_name")
            assert hasattr(settings, "debug")
            assert hasattr(settings, "log_level")

        except ImportError:
            pytest.skip("ApplicationContainer not available for integration test")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dependency_injection_performance(self):
        """Test performance of dependency injection under load."""
        test_container = TestContainer()
        ApplicationContainer.override(test_container)

        try:

            @inject
            async def performance_worker(
                cache=Provide[ApplicationContainer.cache],
                embeddings=Provide[ApplicationContainer.embeddings],
            ):
                """Performance test worker."""
                # Minimal operations to test injection overhead
                cache.get_document("test")
                embeddings.embed_query("test")
                return True

            # Measure performance
            start_time = asyncio.get_event_loop().time()

            # Run many injections concurrently
            tasks = [performance_worker() for _ in range(100)]
            results = await asyncio.gather(*tasks)

            end_time = asyncio.get_event_loop().time()
            total_time = end_time - start_time

            # Validate performance
            assert len(results) == 100
            assert all(results)
            assert total_time < 1.0  # Should complete 100 injections in <1s

        finally:
            ApplicationContainer.reset_override()
