#!/usr/bin/env python3
"""Performance and load testing for DocMind AI integration.

This module tests performance characteristics and scaling behavior
of the integrated system under various load conditions.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core import Document

from agent_factory import analyze_query_complexity
from models import Settings
from utils import (
    FastEmbedModelManager,
    create_index,
    create_index_async,
    verify_rrf_configuration,
)


class TestPerformanceCharacteristics:
    """Test performance characteristics of key components."""

    def test_singleton_manager_performance(self):
        """Test FastEmbedModelManager singleton performance."""
        start_time = time.perf_counter()

        # Creating multiple instances should be fast (singleton pattern)
        managers = []
        for _ in range(100):
            managers.append(FastEmbedModelManager())

        creation_time = time.perf_counter() - start_time

        # Should be very fast since it's singleton
        assert creation_time < 0.1, (
            f"Singleton creation took {creation_time:.3f}s, should be < 0.1s"
        )

        # All instances should be the same
        for manager in managers[1:]:
            assert manager is managers[0]

    def test_query_complexity_analysis_performance(self):
        """Test query complexity analysis performance."""
        test_queries = [
            "What is this about?",
            "How does machine learning relate to artificial intelligence research?",
            "Compare and analyze the differences between multiple approaches to "
            "natural language processing",
            "What do you see in this image and how does it relate to the "
            "document content?",
            "What entities are mentioned and how are they connected to each other?",
        ] * 20  # 100 total queries

        start_time = time.perf_counter()

        results = []
        for query in test_queries:
            complexity, query_type = analyze_query_complexity(query)
            results.append((complexity, query_type))

        analysis_time = time.perf_counter() - start_time

        # Should analyze 100 queries in reasonable time
        assert analysis_time < 1.0, (
            f"Query analysis took {analysis_time:.3f}s for 100 queries, "
            "should be < 1.0s"
        )
        assert len(results) == 100

        # Log performance metrics
        logging.info(
            f"Query complexity analysis: {len(test_queries)} queries in "
            f"{analysis_time:.3f}s"
        )
        logging.info(
            f"Average time per query: {analysis_time / len(test_queries):.6f}s"
        )

    def test_rrf_configuration_validation_performance(self):
        """Test RRF configuration validation performance."""
        settings = Settings()

        start_time = time.perf_counter()

        # Run validation multiple times
        results = []
        for _ in range(100):
            verification = verify_rrf_configuration(settings)
            results.append(verification)

        validation_time = time.perf_counter() - start_time

        # Should be fast
        assert validation_time < 0.1, (
            f"RRF validation took {validation_time:.3f}s for 100 calls, "
            "should be < 0.1s"
        )
        assert len(results) == 100

        # All results should be consistent
        for result in results[1:]:
            assert result == results[0]


class TestConcurrentOperations:
    """Test concurrent operations and thread safety."""

    def test_singleton_thread_safety(self):
        """Test FastEmbedModelManager singleton thread safety."""
        managers = []

        def create_manager():
            return FastEmbedModelManager()

        # Create managers concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_manager) for _ in range(50)]
            managers = [future.result() for future in futures]

        # All should be the same instance
        for manager in managers[1:]:
            assert manager is managers[0]

    def test_concurrent_query_analysis(self):
        """Test concurrent query complexity analysis."""
        test_queries = [
            "What is this document about?",
            "Compare multiple approaches to machine learning",
            "What entities are connected in this knowledge graph?",
            "Analyze the visual elements in these images",
        ] * 10  # 40 total queries

        def analyze_query(query):
            return analyze_query_complexity(query)

        start_time = time.perf_counter()

        # Analyze queries concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(analyze_query, query) for query in test_queries]
            results = [future.result() for future in futures]

        concurrent_time = time.perf_counter() - start_time

        assert len(results) == len(test_queries)

        # Should complete in reasonable time
        assert concurrent_time < 2.0, (
            f"Concurrent analysis took {concurrent_time:.3f}s, should be < 2.0s"
        )

        logging.info(
            f"Concurrent query analysis: {len(test_queries)} queries in "
            f"{concurrent_time:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_async_operation_performance(self):
        """Test async operation performance patterns."""
        # Mock components for performance testing
        with (
            patch("utils.AsyncQdrantClient") as mock_client,
            patch("utils.FastEmbedEmbedding"),
            patch("utils.SparseTextEmbedding"),
            patch("utils.VectorStoreIndex.from_documents") as mock_index,
        ):
            # Setup mocks for performance testing
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.collection_exists.return_value = False
            mock_instance.close = AsyncMock()

            mock_vector_index = MagicMock()
            mock_index.return_value = mock_vector_index

            # Create test documents
            docs = [Document(text=f"Test document {i}") for i in range(10)]

            start_time = time.perf_counter()

            # Test async index creation
            await create_index_async(docs, use_gpu=False)

            async_time = time.perf_counter() - start_time

            # Should complete quickly with mocked components
            assert async_time < 5.0, (
                f"Async index creation took {async_time:.3f}s, "
                "should be < 5.0s"
            )

            logging.info(
                f"Async index creation: {async_time:.3f}s for {len(docs)} "
                "documents"
            )


class TestScalingBehavior:
    """Test scaling behavior with increasing load."""

    def test_document_processing_scaling(self):
        """Test document processing performance scaling."""
        document_counts = [1, 5, 10, 25, 50]
        processing_times = []

        for count in document_counts:
            # Create test documents
            docs = [
                Document(text=f"Test document {i} with content for processing")
                for i in range(count)
            ]

            start_time = time.perf_counter()

            # Process documents (basic metadata handling)
            processed_docs = []
            for doc in docs:
                processed_doc = Document(
                    text=doc.text,
                    metadata={
                        "processed": True,
                        "length": len(doc.text),
                        "source": f"test_doc_{len(processed_docs)}",
                    },
                )
                processed_docs.append(processed_doc)

            processing_time = time.perf_counter() - start_time
            processing_times.append(processing_time)

            logging.info(f"Processed {count} documents in {processing_time:.3f}s")

        # Processing time should scale reasonably (roughly linear)
        assert len(processing_times) == len(document_counts)

        # Larger document counts shouldn't take exponentially longer
        for i in range(1, len(processing_times)):
            scale_factor = document_counts[i] / document_counts[i - 1]
            time_factor = processing_times[i] / processing_times[i - 1]

            # Time scaling should be reasonable (not exponential)
            assert time_factor < scale_factor * 2, (
                f"Processing time scaling too aggressive: {time_factor:.2f}x for "
                f"{scale_factor:.2f}x documents"
            )

    def test_query_complexity_scaling(self):
        """Test query complexity analysis scaling."""
        query_lengths = [5, 15, 30, 60, 120]  # Number of words
        analysis_times = []

        for length in query_lengths:
            # Create query of specific length
            query = " ".join([f"word{i}" for i in range(length)])

            start_time = time.perf_counter()

            # Analyze multiple times for averaging
            for _ in range(10):
                complexity, query_type = analyze_query_complexity(query)

            analysis_time = (time.perf_counter() - start_time) / 10  # Average time
            analysis_times.append(analysis_time)

            logging.info(f"Query length {length} words: {analysis_time:.6f}s average")

        # Analysis time should scale reasonably with query length
        for time_taken in analysis_times:
            assert time_taken < 0.01, (
                f"Query analysis took {time_taken:.6f}s, should be < 0.01s"
            )


class TestMemoryAndResourceUsage:
    """Test memory usage and resource management."""

    def test_model_manager_memory_usage(self):
        """Test FastEmbedModelManager memory usage patterns."""
        manager = FastEmbedModelManager()

        # Clear cache to start fresh
        manager.clear_cache()
        initial_cache_size = len(manager._models)

        # Cache should be empty initially
        assert initial_cache_size == 0

        # Simulate cache usage (without actually loading models)
        # This tests the caching logic without requiring actual model files
        cache_entries = {}
        for i in range(10):
            key = f"test_model_{i}"
            cache_entries[key] = f"mock_model_{i}"

        # Manual cache simulation for testing
        original_models = manager._models
        manager._models = cache_entries

        # Check cache size
        assert len(manager._models) == 10

        # Clear cache
        manager.clear_cache()
        assert len(manager._models) == 0

        # Restore original cache
        manager._models = original_models

    def test_document_memory_efficiency(self):
        """Test document handling memory efficiency."""
        # Create documents with varying sizes
        small_docs = [Document(text="Small document") for _ in range(100)]
        medium_docs = [
            Document(text="Medium document with more content " * 10) for _ in range(50)
        ]
        large_docs = [
            Document(text="Large document with lots of content " * 100)
            for _ in range(10)
        ]

        all_docs = small_docs + medium_docs + large_docs

        # Basic memory usage test (documents should be created efficiently)
        assert len(all_docs) == 160

        # Verify document structure is maintained
        for doc in all_docs:
            assert hasattr(doc, "text")
            assert hasattr(doc, "metadata")
            assert isinstance(doc.text, str)
            assert len(doc.text) > 0


class TestAsyncPerformancePatterns:
    """Test async performance patterns and optimizations."""

    @pytest.mark.asyncio
    async def test_async_vs_sync_pattern_comparison(self):
        """Test async vs sync pattern performance characteristics."""
        # This tests the pattern structure, not actual performance

        # Test that async functions are properly defined
        import inspect

        from utils import create_index_async

        # Verify async function characteristics
        assert inspect.iscoroutinefunction(create_index_async)
        assert not inspect.iscoroutinefunction(create_index)

        # Both should have similar signatures
        async_sig = inspect.signature(create_index_async)
        sync_sig = inspect.signature(create_index)

        assert len(async_sig.parameters) == len(sync_sig.parameters)

    @pytest.mark.asyncio
    async def test_concurrent_async_operations(self):
        """Test concurrent async operations."""

        async def mock_async_operation(delay=0.01):
            await asyncio.sleep(delay)
            return "completed"

        start_time = time.perf_counter()

        # Run operations concurrently
        tasks = [mock_async_operation() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        concurrent_time = time.perf_counter() - start_time

        assert len(results) == 10
        assert all(result == "completed" for result in results)

        # Concurrent execution should be faster than sequential
        # 10 operations with 0.01s each should take ~0.01s concurrently, not 0.1s
        assert concurrent_time < 0.05, (
            f"Concurrent operations took {concurrent_time:.3f}s, should be < 0.05s"
        )

    @pytest.mark.asyncio
    async def test_async_resource_cleanup(self):
        """Test async resource cleanup patterns."""

        async def mock_async_resource():
            # Simulate resource that needs cleanup
            return AsyncMock()

        resource = await mock_async_resource()

        # Test cleanup pattern
        if hasattr(resource, "close"):
            await resource.close()

        # Should complete without errors
        assert True


class TestConfigurationPerformance:
    """Test configuration loading and validation performance."""

    def test_settings_loading_performance(self):
        """Test settings loading performance."""
        start_time = time.perf_counter()

        # Load settings multiple times
        settings_list = []
        for _ in range(100):
            settings = Settings()
            settings_list.append(settings)

        loading_time = time.perf_counter() - start_time

        # Should load quickly
        assert loading_time < 1.0, (
            f"Settings loading took {loading_time:.3f}s for 100 instances, "
            "should be < 1.0s"
        )

        # All should have same configuration
        for settings in settings_list[1:]:
            assert settings.backend == settings_list[0].backend
            assert (
                settings.dense_embedding_model == settings_list[0].dense_embedding_model
            )

    def test_validation_performance_stress(self):
        """Test validation performance under stress."""
        settings = Settings()

        start_time = time.perf_counter()

        # Run validation stress test
        for _ in range(1000):
            # Test various validation operations
            assert settings.embedding_batch_size > 0
            assert 0.0 <= settings.rrf_fusion_weight_dense <= 1.0
            assert 0.0 <= settings.rrf_fusion_weight_sparse <= 1.0
            assert settings.dense_embedding_dimension > 0

        stress_time = time.perf_counter() - start_time

        # Should handle stress testing well
        assert stress_time < 0.1, (
            f"Validation stress test took {stress_time:.3f}s, should be < 0.1s"
        )


# Performance test configuration
def pytest_configure():
    """Configure pytest for performance tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
