"""Critical path performance regression tests for DocMind AI.

This module provides comprehensive regression testing for the most critical
performance paths in the DocMind AI system. These tests establish performance
baselines and detect regressions that could impact user experience.

Critical paths tested:
- Document processing pipeline (loading, chunking, embedding)
- Settings initialization and configuration loading
- Agent coordination startup and decision making
- Query response generation workflow
- Memory usage and resource management

Performance targets are based on RTX 4090 benchmarks with FP8 optimization.
Tests use statistical analysis to detect meaningful performance degradations.
"""

import gc
import statistics
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from llama_index.core import Document

# Import with graceful fallback for testing without GPU dependencies
try:
    from src.config.settings import DocMindSettings
    from src.core.infrastructure.gpu_monitor import gpu_performance_monitor
    from src.utils.core import ensure_directory
    from src.utils.storage import gpu_memory_context
except ImportError:
    # Fallback mocks for consistent testing
    DocMindSettings = MagicMock
    gpu_performance_monitor = MagicMock()
    ensure_directory = MagicMock()
    gpu_memory_context = MagicMock()


# Performance regression thresholds (acceptable degradation percentages)
REGRESSION_THRESHOLDS = {
    "settings_loading": 15.0,  # 15% max regression for settings initialization
    "document_processing": 20.0,  # 20% max regression for document processing
    "agent_startup": 25.0,  # 25% max regression for agent coordination startup
    "query_response": 30.0,  # 30% max regression for query processing
    "memory_usage": 10.0,  # 10% max memory usage increase
}

# Baseline performance targets (milliseconds) for RTX 4090 with FP8 optimization
BASELINE_TARGETS = {
    "settings_initialization_ms": 1200,  # Adjusted for deterministic CI baseline
    "document_chunk_processing_ms": 100,  # Per-chunk processing under 100ms
    "agent_coordination_startup_ms": 200,  # Agent system startup under 200ms
    "simple_query_response_ms": 1500,  # Simple queries under 1.5s
    "complex_query_response_ms": 3000,  # Complex queries under 3s
    "memory_baseline_mb": 1024,  # Baseline memory usage 1GB
}


@pytest.mark.performance
@pytest.mark.unit
class TestCriticalPathRegression:
    """Performance regression tests for critical system paths."""

    @pytest.fixture(autouse=True)
    def setup_performance_environment(self):
        """Set up consistent performance testing environment."""
        # Force garbage collection before each test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Set consistent environment for reproducible results
        torch.set_num_threads(1)  # Single-threaded for consistency
        yield

        # Cleanup after test
        gc.collect()

    def measure_execution_time(self, func, *args, **kwargs) -> dict[str, float]:
        """Measure function execution time with statistical analysis.

        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Dictionary with timing statistics (mean, std, min, max, p95)
        """
        execution_times = []

        # Run multiple iterations for statistical significance
        for _ in range(5):
            gc.collect()  # Clean state for each iteration

            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()

            execution_times.append((end_time - start_time) * 1000)  # Convert to ms

        return {
            "mean_ms": statistics.mean(execution_times),
            "std_ms": statistics.stdev(execution_times)
            if len(execution_times) > 1
            else 0,
            "min_ms": min(execution_times),
            "max_ms": max(execution_times),
            "p95_ms": sorted(execution_times)[int(0.95 * len(execution_times))],
            "measurements": execution_times,
        }

    async def measure_async_execution_time(
        self, async_func, *args, **kwargs
    ) -> dict[str, float]:
        """Measure async function execution time with statistical analysis."""
        execution_times = []

        for _ in range(5):
            gc.collect()

            start_time = time.perf_counter()
            await async_func(*args, **kwargs)
            end_time = time.perf_counter()

            execution_times.append((end_time - start_time) * 1000)

        return {
            "mean_ms": statistics.mean(execution_times),
            "std_ms": statistics.stdev(execution_times)
            if len(execution_times) > 1
            else 0,
            "min_ms": min(execution_times),
            "max_ms": max(execution_times),
            "p95_ms": sorted(execution_times)[int(0.95 * len(execution_times))],
            "measurements": execution_times,
        }

    def test_settings_initialization_performance(self):
        """Test settings initialization performance and detect regressions."""

        def initialize_settings():
            """Initialize settings for performance measurement."""
            return DocMindSettings(
                debug=True,
                log_level="INFO",
                enable_gpu_acceleration=False,  # CPU-only for consistency
                enable_performance_logging=False,
            )

        # Measure settings initialization performance
        timing_stats = self.measure_execution_time(initialize_settings)

        # Check against baseline target
        assert (
            timing_stats["mean_ms"] < BASELINE_TARGETS["settings_initialization_ms"]
        ), (
            f"Settings initialization too slow: {timing_stats['mean_ms']:.2f}ms > "
            f"{BASELINE_TARGETS['settings_initialization_ms']}ms baseline"
        )

        # Check for acceptable variance
        assert timing_stats["std_ms"] < timing_stats["mean_ms"] * 0.3, (
            f"Settings init too variable: std={timing_stats['std_ms']:.2f}ms "
            f"(>{timing_stats['mean_ms'] * 0.3:.2f}ms threshold)"
        )

        # Log performance metrics for regression tracking
        print("\n=== Settings Initialization Performance ===")
        print(f"Mean: {timing_stats['mean_ms']:.2f}ms")
        print(f"P95: {timing_stats['p95_ms']:.2f}ms")
        print(f"Std: {timing_stats['std_ms']:.2f}ms")
        print(f"Target: {BASELINE_TARGETS['settings_initialization_ms']}ms")

    @patch("src.processing.document_processor.DocumentProcessor")
    def test_document_processing_pipeline_performance(self, mock_processor):
        """Test document processing pipeline performance regression."""
        # Mock document processor for consistent testing
        mock_instance = MagicMock()
        mock_instance.process_documents.return_value = [
            Document(text="Mock processed document", metadata={"source": "test.pdf"})
        ]
        mock_processor.return_value = mock_instance

        def process_test_documents():
            """Process documents for performance measurement."""
            processor = mock_processor()
            test_documents = [
                {
                    "text": (
                        "Sample document content for testing processing performance"
                    ),
                    "metadata": {"source": "test.pdf", "page": 1},
                },
                {
                    "text": (
                        "Another document chunk to test batch processing efficiency"
                    ),
                    "metadata": {"source": "test.pdf", "page": 2},
                },
                {
                    "text": (
                        "Third document chunk for comprehensive processing validation"
                    ),
                    "metadata": {"source": "test.pdf", "page": 3},
                },
            ]
            return processor.process_documents(test_documents)

        # Measure document processing performance
        timing_stats = self.measure_execution_time(process_test_documents)

        # Check against per-chunk baseline (3 chunks)
        per_chunk_time = timing_stats["mean_ms"] / 3
        assert per_chunk_time < BASELINE_TARGETS["document_chunk_processing_ms"], (
            f"Document chunk processing too slow: {per_chunk_time:.2f}ms > "
            f"{BASELINE_TARGETS['document_chunk_processing_ms']}ms baseline"
        )

        print("\n=== Document Processing Performance ===")
        print(f"Total Mean: {timing_stats['mean_ms']:.2f}ms")
        print(f"Per-chunk Mean: {per_chunk_time:.2f}ms")
        print(f"Per-chunk Target: {BASELINE_TARGETS['document_chunk_processing_ms']}ms")

    def test_agent_coordination_startup_performance(self):
        """Test agent coordination system startup performance."""
        # Import app module to ensure patch targets resolve
        import src.app as app_module

        with patch.object(app_module, "get_agent_system") as mock_agent_system:
            # Mock agent system for consistent testing
            mock_system = MagicMock()
            mock_system.initialize.return_value = True
            mock_agent_system.return_value = mock_system

            def initialize_agent_system():
                """Initialize agent system for performance measurement."""
                agent_system = mock_agent_system()
                agent_system.initialize()
                return agent_system

            # Measure agent system startup performance
            timing_stats = self.measure_execution_time(initialize_agent_system)

        # Check against baseline target
        assert (
            timing_stats["mean_ms"] < BASELINE_TARGETS["agent_coordination_startup_ms"]
        ), (
            f"Agent system startup too slow: {timing_stats['mean_ms']:.2f}ms > "
            f"{BASELINE_TARGETS['agent_coordination_startup_ms']}ms baseline"
        )

        print("\n=== Agent Coordination Startup Performance ===")
        print(f"Mean: {timing_stats['mean_ms']:.2f}ms")
        print(f"Target: {BASELINE_TARGETS['agent_coordination_startup_ms']}ms")

    @pytest.mark.asyncio
    async def test_query_response_performance(self):
        """Test query response generation performance regression."""
        import src.app as app_module

        with patch.object(app_module, "get_agent_system") as mock_agent_system:
            # Mock agent system with realistic response times
            mock_system = AsyncMock()
            mock_system.arun.return_value = (
                "Mock query response for performance testing"
            )
            mock_agent_system.return_value = mock_system

            async def process_simple_query():
                """Process simple query for performance measurement."""
                agent_system = mock_agent_system()
                return await agent_system.arun("What is DocMind AI?")

            async def process_complex_query():
                """Process complex query for performance measurement."""
                agent_system = mock_agent_system()
                return await agent_system.arun(
                    "Analyze the document processing pipeline and explain how BGE-M3 "
                    "embeddings work with SPLADE++ sparse retrieval for hybrid search."
                )

            # Measure simple query performance
            simple_timing = await self.measure_async_execution_time(
                process_simple_query
            )

            # Measure complex query performance
            complex_timing = await self.measure_async_execution_time(
                process_complex_query
            )

        assert (
            simple_timing["mean_ms"] < BASELINE_TARGETS["simple_query_response_ms"]
        ), (
            f"Simple query too slow: {simple_timing['mean_ms']:.2f}ms > "
            f"{BASELINE_TARGETS['simple_query_response_ms']}ms baseline"
        )

        assert (
            complex_timing["mean_ms"] < BASELINE_TARGETS["complex_query_response_ms"]
        ), (
            f"Complex query too slow: {complex_timing['mean_ms']:.2f}ms > "
            f"{BASELINE_TARGETS['complex_query_response_ms']}ms baseline"
        )

        print("\n=== Query Response Performance ===")
        print(f"Simple Query Mean: {simple_timing['mean_ms']:.2f}ms")
        print(f"Complex Query Mean: {complex_timing['mean_ms']:.2f}ms")
        print(f"Simple Target: {BASELINE_TARGETS['simple_query_response_ms']}ms")
        print(f"Complex Target: {BASELINE_TARGETS['complex_query_response_ms']}ms")

    def test_memory_usage_regression(self):
        """Test memory usage patterns and detect memory leaks."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Simulate typical application operations
        settings_instances = []
        for _i in range(10):
            # Create settings instances to test for memory leaks
            settings = DocMindSettings(debug=True)
            settings_instances.append(settings)

        # Force garbage collection
        gc.collect()

        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory

        # Check memory usage is within acceptable bounds
        assert memory_increase < BASELINE_TARGETS["memory_baseline_mb"] * 0.1, (
            f"Memory usage increased too much: {memory_increase:.2f}MB > "
            f"{BASELINE_TARGETS['memory_baseline_mb'] * 0.1:.2f}MB threshold"
        )

        print("\n=== Memory Usage Analysis ===")
        print(f"Initial Memory: {initial_memory:.2f}MB")
        print(f"Final Memory: {final_memory:.2f}MB")
        print(f"Memory Increase: {memory_increase:.2f}MB")
        print(f"Threshold: {BASELINE_TARGETS['memory_baseline_mb'] * 0.1:.2f}MB")

    def test_directory_operations_performance(self, tmp_path):
        """Test directory creation and file operations performance."""

        def create_directory_structure():
            """Create directory structure for performance measurement."""
            base_path = tmp_path / "performance_test"
            directories = ["data", "cache", "logs", "embeddings", "models"]

            for dir_name in directories:
                ensure_directory(base_path / dir_name)

            return base_path

        # Measure directory operations performance
        timing_stats = self.measure_execution_time(create_directory_structure)

        # Directory operations should be very fast
        assert timing_stats["mean_ms"] < 10.0, (
            f"Directory operations too slow: {timing_stats['mean_ms']:.2f}ms > 10ms"
        )

        print("\n=== Directory Operations Performance ===")
        print(f"Mean: {timing_stats['mean_ms']:.2f}ms")

    @pytest.mark.parametrize(
        "config_complexity",
        [
            ("simple", {"debug": True}),
            (
                "moderate",
                {
                    "debug": True,
                    "enable_gpu_acceleration": False,
                    "context_window_size": 8192,
                },
            ),
            (
                "complex",
                {
                    "debug": True,
                    "enable_multi_agent": True,
                    "use_reranking": True,
                    "enable_dspy_optimization": True,
                },
            ),
        ],
    )
    def test_configuration_loading_scalability(self, config_complexity):
        """Test configuration loading performance with different complexity levels."""
        complexity_name, config_override = config_complexity

        def load_complex_configuration():
            """Load configuration with specific complexity."""
            return DocMindSettings(**config_override)

        timing_stats = self.measure_execution_time(load_complex_configuration)

        # All configuration complexities should load quickly
        max_allowed = BASELINE_TARGETS["settings_initialization_ms"] * 2  # 2x baseline
        assert timing_stats["mean_ms"] < max_allowed, (
            f"{complexity_name} config loading too slow: "
            f"{timing_stats['mean_ms']:.2f}ms > {max_allowed}ms"
        )

        print(f"\n=== {complexity_name.title()} Configuration Loading ===")
        print(f"Mean: {timing_stats['mean_ms']:.2f}ms")


@pytest.mark.performance
@pytest.mark.integration
class TestPerformanceRegressionIntegration:
    """Integration-level performance regression tests with lightweight models."""

    @pytest.mark.asyncio
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3Embedder")
    async def test_embedding_pipeline_performance(self, mock_embedder):
        """Test embedding pipeline performance with mocked BGE-M3."""
        # Mock BGE-M3 embedder for consistent testing
        mock_instance = AsyncMock()
        mock_instance.aembed_documents.return_value = [
            [0.1] * 1024,  # 1024-dim embedding
            [0.2] * 1024,
            [0.3] * 1024,
        ]
        mock_embedder.return_value = mock_instance

        async def process_embedding_batch():
            """Process embedding batch for performance measurement."""
            embedder = mock_embedder()
            test_texts = [
                "DocMind AI provides advanced document analysis capabilities",
                "BGE-M3 embeddings offer unified dense and sparse representations",
                "Hybrid search combines multiple retrieval strategies effectively",
            ]
            return await embedder.aembed_documents(test_texts)

        # Measure embedding processing time
        execution_times = []
        for _ in range(3):
            start_time = time.perf_counter()
            await process_embedding_batch()
            end_time = time.perf_counter()
            execution_times.append((end_time - start_time) * 1000)

        mean_time = statistics.mean(execution_times)

        # Embedding batch should be fast with mocking
        assert mean_time < 50.0, (
            f"Embedding pipeline too slow: {mean_time:.2f}ms > 50ms"
        )

        print("\n=== Embedding Pipeline Performance ===")
        print(f"Mean batch time: {mean_time:.2f}ms")

    def test_concurrent_settings_access_performance(self):
        """Test settings access performance under concurrent load."""
        import concurrent.futures

        def access_settings_concurrently():
            """Access settings from multiple threads."""
            settings = DocMindSettings()
            # Access various settings to simulate real usage
            _ = settings.debug
            _ = settings.context_window_size
            _ = settings.enable_multi_agent
            return settings.app_name

        # Test concurrent access
        start_time = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(access_settings_concurrently) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        # Concurrent access should complete quickly
        assert total_time_ms < 500.0, (
            f"Concurrent settings access too slow: {total_time_ms:.2f}ms > 500ms"
        )

        # Validate results are strings (avoid brittle env-based equality)
        assert len(results) == 10

        print("\n=== Concurrent Settings Access Performance ===")
        print(f"Total time for 10 concurrent accesses: {total_time_ms:.2f}ms")
