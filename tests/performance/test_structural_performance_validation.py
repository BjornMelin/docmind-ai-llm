"""Structural Changes Performance Validation Tests.

This module provides comprehensive performance testing to validate that the extensive
structural improvements haven't introduced regressions:

- Import performance testing after directory flattening (6 levels → 2 levels)
- Configuration loading performance with unified Pydantic system
- Memory footprint validation across reorganized modules
- Cross-module dependency performance analysis
- Resource management efficiency after factory pattern changes

Establishes performance baselines post-cleanup for regression detection.
"""

import gc
import importlib
import sys
import time
from typing import Any
from unittest.mock import patch

import psutil
import pytest

from src.config.settings import DocMindSettings

# Performance benchmarks for structural changes validation
IMPORT_PERFORMANCE_TARGETS = {
    "single_module_import_ms": 100,  # Individual module imports should be <100ms
    "config_loading_ms": 50,  # Configuration loading should be <50ms
    "total_core_imports_ms": 500,  # All core modules should load in <500ms
    "memory_overhead_mb": 50,  # Additional memory overhead should be <50MB
}

MEMORY_BASELINE_TOLERANCE_MB = 10  # Allow 10MB variance in memory measurements


@pytest.fixture
def performance_tracker():
    """Fixture for tracking import and configuration performance."""

    class StructuralPerformanceTracker:
        def __init__(self):
            self.measurements = {}
            self.memory_baselines = {}
            self.import_times = {}

        def measure_import_time(self, module_name: str) -> float:
            """Measure time to import a module."""
            # Clear module from cache if already imported
            if module_name in sys.modules:
                del sys.modules[module_name]

            start_time = time.perf_counter()
            try:
                importlib.import_module(module_name)
                import_time_ms = (time.perf_counter() - start_time) * 1000
                self.import_times[module_name] = import_time_ms
                return import_time_ms
            except ImportError as e:
                # Record failed imports for analysis
                self.import_times[module_name] = {"error": str(e), "time_ms": None}
                raise

        def measure_memory_baseline(self, operation_name: str) -> dict[str, float]:
            """Measure memory usage before operation."""
            gc.collect()  # Clean up before measurement
            process = psutil.Process()

            baseline = {
                "rss_mb": process.memory_info().rss / (1024 * 1024),
                "vms_mb": process.memory_info().vms / (1024 * 1024),
                "timestamp": time.perf_counter(),
            }

            self.memory_baselines[operation_name] = baseline
            return baseline

        def measure_memory_delta(self, operation_name: str) -> dict[str, float]:
            """Measure memory change since baseline."""
            if operation_name not in self.memory_baselines:
                raise ValueError(f"No baseline found for operation: {operation_name}")

            gc.collect()
            process = psutil.Process()
            baseline = self.memory_baselines[operation_name]

            current = {
                "rss_mb": process.memory_info().rss / (1024 * 1024),
                "vms_mb": process.memory_info().vms / (1024 * 1024),
                "timestamp": time.perf_counter(),
            }

            delta = {
                "rss_delta_mb": current["rss_mb"] - baseline["rss_mb"],
                "vms_delta_mb": current["vms_mb"] - baseline["vms_mb"],
                "duration_ms": (current["timestamp"] - baseline["timestamp"]) * 1000,
            }

            self.measurements[operation_name] = delta
            return delta

        def benchmark_operation(
            self, operation_name: str, operation_func, *args, **kwargs
        ):
            """Benchmark an operation with timing and memory tracking."""
            self.measure_memory_baseline(operation_name)

            start_time = time.perf_counter()
            try:
                result = operation_func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000

                # memory_delta = self.measure_memory_delta(operation_name)
                # NOTE: Memory delta not used in current measurements

                self.measurements[operation_name].update(
                    {"duration_ms": duration_ms, "success": True, "result": result}
                )

                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.measurements[operation_name] = {
                    "duration_ms": duration_ms,
                    "success": False,
                    "error": str(e),
                }
                raise

        def validate_performance_targets(
            self, targets: dict[str, float]
        ) -> dict[str, Any]:
            """Validate measurements against performance targets."""
            validation_results = {
                "targets_met": True,
                "failed_targets": [],
                "results": {},
            }

            for target_name, target_value in targets.items():
                if target_name.endswith("_import_ms"):
                    # Check import time targets
                    module_prefix = target_name.replace("_import_ms", "")
                    matching_imports = {
                        k: v
                        for k, v in self.import_times.items()
                        if module_prefix in k and isinstance(v, int | float)
                    }

                    if matching_imports:
                        actual_value = max(matching_imports.values())
                    else:
                        continue

                elif target_name.endswith("_ms"):
                    # Check operation duration targets
                    operation_name = target_name.replace("_ms", "")
                    if operation_name in self.measurements:
                        actual_value = self.measurements[operation_name].get(
                            "duration_ms", 0
                        )
                    else:
                        continue

                elif target_name.endswith("_mb"):
                    # Check memory usage targets
                    operation_name = target_name.replace("_mb", "")
                    if operation_name in self.measurements:
                        actual_value = self.measurements[operation_name].get(
                            "rss_delta_mb", 0
                        )
                    else:
                        continue
                else:
                    continue

                validation_results["results"][target_name] = {
                    "target": target_value,
                    "actual": actual_value,
                    "met": actual_value <= target_value,
                    "margin": (actual_value - target_value) / target_value
                    if target_value > 0
                    else 0.0,
                }

                if actual_value > target_value:
                    validation_results["targets_met"] = False
                    validation_results["failed_targets"].append(target_name)

            return validation_results

    return StructuralPerformanceTracker()


@pytest.mark.performance
class TestImportPerformancePostFlattening:
    """Test import performance after directory structure flattening."""

    def test_core_module_import_performance(self, performance_tracker):
        """Test that flattened core modules import within performance targets."""
        # Core modules that should import quickly after flattening
        core_modules = [
            "src.config.settings",
            "src.models.processing",
            "src.models.embeddings",
            "src.models.storage",
            "src.utils.core",
            "src.utils.document",
            "src.utils.storage",
            "src.retrieval.embeddings",
            "src.retrieval.vector_store",
            "src.retrieval.query_engine",
            "src.agents.coordinator",
            "src.agents.tools",
        ]

        import_results = {}
        total_import_time = 0

        for module in core_modules:
            try:
                import_time_ms = performance_tracker.measure_import_time(module)
                import_results[module] = import_time_ms
                total_import_time += import_time_ms

                # Individual modules should import quickly
                assert (
                    import_time_ms
                    <= IMPORT_PERFORMANCE_TARGETS["single_module_import_ms"]
                ), (
                    f"Module {module} imports too slowly: {import_time_ms:.1f}ms > "
                    f"{IMPORT_PERFORMANCE_TARGETS['single_module_import_ms']}ms"
                )

            except ImportError as e:
                pytest.fail(f"Core module {module} failed to import: {e}")

        # Total import time for all core modules should be reasonable
        assert (
            total_import_time <= IMPORT_PERFORMANCE_TARGETS["total_core_imports_ms"]
        ), (
            f"Total core import time too high: {total_import_time:.1f}ms > "
            f"{IMPORT_PERFORMANCE_TARGETS['total_core_imports_ms']}ms"
        )

        print(f"Import performance results: {import_results}")
        print(f"Total core imports time: {total_import_time:.1f}ms")

    def test_no_circular_import_delays(self, performance_tracker):
        """Test that circular import prevention doesn't introduce delays."""
        # Test imports that could have circular dependencies
        potentially_circular_imports = [
            ("src.config.settings", "src.utils.core"),
            ("src.retrieval.embeddings", "src.models.embeddings"),
            ("src.agents.coordinator", "src.agents.tools"),
            ("src.utils.document", "src.processing.document_processor"),
        ]

        for module_a, module_b in potentially_circular_imports:
            # Import both modules and measure total time
            performance_tracker.measure_memory_baseline(
                f"circular_test_{module_a}_{module_b}"
            )

            start_time = time.perf_counter()

            try:
                # Clear modules from cache
                if module_a in sys.modules:
                    del sys.modules[module_a]
                if module_b in sys.modules:
                    del sys.modules[module_b]

                # Import both modules
                importlib.import_module(module_a)
                importlib.import_module(module_b)

                total_time_ms = (time.perf_counter() - start_time) * 1000

                # Should not take excessive time due to circular resolution
                assert total_time_ms <= 200, (
                    f"Potential circular import delay between {module_a} "
                    f"and {module_b}: {total_time_ms:.1f}ms"
                )

                print(
                    f"No circular delay: {module_a} + {module_b} = "
                    f"{total_time_ms:.1f}ms"
                )

            except ImportError as e:
                pytest.fail(f"Failed to import modules {module_a}, {module_b}: {e}")

            performance_tracker.measure_memory_delta(
                f"circular_test_{module_a}_{module_b}"
            )

    def test_import_memory_overhead(self, performance_tracker):
        """Test that flattened imports don't introduce memory overhead."""
        # Measure memory before any imports
        # baseline_modules = list(sys.modules.keys())
        # NOTE: baseline_modules not used - test focuses on memory tracker baseline
        performance_tracker.measure_memory_baseline("import_memory_test")

        # Import core modules and measure memory impact
        modules_to_test = [
            "src.config.settings",
            "src.models.processing",
            "src.retrieval.embeddings",
            "src.agents.coordinator",
        ]

        for module in modules_to_test:
            if module not in sys.modules:
                importlib.import_module(module)

        # Measure memory overhead
        memory_delta = performance_tracker.measure_memory_delta("import_memory_test")

        # Memory overhead should be reasonable
        assert (
            memory_delta["rss_delta_mb"]
            <= IMPORT_PERFORMANCE_TARGETS["memory_overhead_mb"]
        ), (
            f"Import memory overhead too high: {memory_delta['rss_delta_mb']:.1f}MB > "
            f"{IMPORT_PERFORMANCE_TARGETS['memory_overhead_mb']}MB"
        )

        print(f"Import memory overhead: {memory_delta['rss_delta_mb']:.1f}MB")


@pytest.mark.performance
class TestUnifiedConfigurationPerformance:
    """Test performance of unified Pydantic configuration system."""

    def test_configuration_loading_speed(self, performance_tracker):
        """Test that unified configuration loads within performance targets."""

        def load_settings():
            return DocMindSettings()

        # Benchmark configuration loading
        performance_tracker.benchmark_operation("config_loading", load_settings)

        config_result = performance_tracker.measurements["config_loading"]

        # Configuration loading should be fast
        assert (
            config_result["duration_ms"]
            <= IMPORT_PERFORMANCE_TARGETS["config_loading_ms"]
        ), (
            f"Configuration loading too slow: {config_result['duration_ms']:.1f}ms > "
            f"{IMPORT_PERFORMANCE_TARGETS['config_loading_ms']}ms"
        )

        # Should not consume excessive memory
        assert config_result["rss_delta_mb"] <= 20, (
            f"Configuration memory usage too high: "
            f"{config_result['rss_delta_mb']:.1f}MB"
        )

        print(
            f"Configuration loading: {config_result['duration_ms']:.1f}ms, "
            f"{config_result['rss_delta_mb']:.1f}MB"
        )

    def test_environment_variable_processing_speed(self, performance_tracker):
        """Test that environment variable processing is efficient."""
        import os
        from unittest.mock import patch

        # Set up test environment variables
        test_env_vars = {
            "DOCMIND_DEBUG": "true",
            "DOCMIND_LOG_LEVEL": "DEBUG",
            "DOCMIND_CHUNK_SIZE": "1024",
            "DOCMIND_TOP_K": "15",
            "DOCMIND_EMBEDDING__MODEL_NAME": "BAAI/bge-m3",
            "DOCMIND_VLLM__GPU_MEMORY_UTILIZATION": "0.8",
            "DOCMIND_AGENTS__ENABLE_MULTI_AGENT": "true",
        }

        def load_settings_with_env():
            with patch.dict(os.environ, test_env_vars):
                return DocMindSettings()

        # Benchmark environment variable processing
        performance_tracker.benchmark_operation(
            "env_processing", load_settings_with_env
        )

        env_result = performance_tracker.measurements["env_processing"]
        settings = env_result["result"]

        # Environment processing should be efficient
        assert (
            env_result["duration_ms"]
            <= IMPORT_PERFORMANCE_TARGETS["config_loading_ms"] * 1.5
        ), f"Environment processing too slow: {env_result['duration_ms']:.1f}ms"

        # Verify environment variables were processed correctly
        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.processing.chunk_size == 1024
        assert settings.retrieval.top_k == 15
        assert settings.embedding.model_name == "BAAI/bge-m3"
        assert settings.vllm.gpu_memory_utilization == 0.8
        assert settings.agents.enable_multi_agent is True

        print(f"Environment processing: {env_result['duration_ms']:.1f}ms")

    def test_nested_model_synchronization_performance(self, performance_tracker):
        """Test that nested model synchronization doesn't introduce delays."""

        def create_and_sync_settings():
            settings = DocMindSettings(
                chunk_size=2048,
                agent_decision_timeout=500,
                bge_m3_model_name="custom-model",
                vllm_gpu_memory_utilization=0.9,
            )
            # Synchronization happens automatically in model_post_init
            # No explicit sync needed
            return settings

        # Benchmark nested model sync
        performance_tracker.benchmark_operation("nested_sync", create_and_sync_settings)

        sync_result = performance_tracker.measurements["nested_sync"]
        settings = sync_result["result"]

        # Synchronization should be fast
        assert sync_result["duration_ms"] <= 20, (
            f"Nested model sync too slow: {sync_result['duration_ms']:.1f}ms"
        )

        # Verify synchronization worked correctly
        assert settings.processing.chunk_size == 2048
        assert settings.agents.decision_timeout == 500
        assert settings.embedding.model_name == "custom-model"
        assert settings.vllm.gpu_memory_utilization == 0.9

        print(f"Nested model sync: {sync_result['duration_ms']:.1f}ms")

    def test_configuration_method_performance(self, performance_tracker):
        """Test that configuration helper methods are efficient."""
        settings = DocMindSettings()

        # Test various configuration method performance
        methods_to_test = [
            ("get_vllm_env_vars", lambda: settings.get_vllm_env_vars()),
            ("get_model_config", lambda: settings.get_model_config()),
            ("get_agent_config", lambda: settings.get_agent_config()),
            ("get_performance_config", lambda: settings.get_performance_config()),
            ("get_embedding_config", lambda: settings.get_embedding_config()),
        ]

        for method_name, method_func in methods_to_test:
            performance_tracker.benchmark_operation(method_name, method_func)

            method_result = performance_tracker.measurements[method_name]

            # Configuration methods should be very fast
            assert method_result["duration_ms"] <= 5, (
                f"{method_name} too slow: {method_result['duration_ms']:.2f}ms"
            )

            # Should return valid configuration
            assert isinstance(method_result["result"], dict)
            assert len(method_result["result"]) > 0

            print(f"{method_name}: {method_result['duration_ms']:.2f}ms")


@pytest.mark.performance
class TestModuleIntegrationPerformance:
    """Test integration performance across reorganized modules."""

    def test_cross_module_import_dependencies(self, performance_tracker):
        """Test that cross-module dependencies work efficiently after reorganization."""
        # Test common integration patterns
        integration_tests = [
            (
                "settings_to_embeddings",
                lambda: self._test_settings_embeddings_integration(),
            ),
            (
                "coordinator_to_tools",
                lambda: self._test_coordinator_tools_integration(),
            ),
            ("utils_to_processing", lambda: self._test_utils_processing_integration()),
            ("models_to_storage", lambda: self._test_models_storage_integration()),
        ]

        for test_name, test_func in integration_tests:
            try:
                performance_tracker.benchmark_operation(test_name, test_func)

                result = performance_tracker.measurements[test_name]

                # Integration should be fast
                assert result["duration_ms"] <= 100, (
                    f"Integration test {test_name} too slow: "
                    f"{result['duration_ms']:.1f}ms"
                )

                # Should use reasonable memory
                assert result["rss_delta_mb"] <= 20, (
                    f"Integration test {test_name} uses too much memory: "
                    f"{result['rss_delta_mb']:.1f}MB"
                )

                print(f"Integration {test_name}: {result['duration_ms']:.1f}ms")

            except Exception as e:
                pytest.fail(f"Integration test {test_name} failed: {e}")

    def _test_settings_embeddings_integration(self):
        """Test settings to embeddings integration."""
        from src.config import settings

        # Test that settings work with embedding configuration
        embedding_config = settings.get_embedding_config()
        assert isinstance(embedding_config, dict)
        assert "model_name" in embedding_config

        return embedding_config

    def _test_coordinator_tools_integration(self):
        """Test coordinator to tools integration."""
        with patch("src.agents.tools.AgentTool"):
            from src.agents.tools import get_available_tools

            # Mock the coordinator and tools integration
            mock_tools = get_available_tools()
            assert isinstance(mock_tools, list | dict)

            return mock_tools

    def _test_utils_processing_integration(self):
        """Test utils to processing integration."""
        from src.processing.document_processor import DocumentProcessor
        from src.utils.document import DocumentLoader

        # Test that document utilities work with processing
        loader = DocumentLoader()
        processor = DocumentProcessor()

        assert loader is not None
        assert processor is not None

        return {"loader": loader, "processor": processor}

    def _test_models_storage_integration(self):
        """Test models to storage integration."""
        from src.models.storage import StorageConfig
        from src.storage.hybrid_persistence import HybridPersistence

        # Test that models work with storage
        storage_config = StorageConfig()
        persistence = HybridPersistence()

        assert storage_config is not None
        assert persistence is not None

        return {"config": storage_config, "persistence": persistence}

    @pytest.mark.asyncio
    async def test_async_integration_performance(self, performance_tracker):
        """Test async integrations work efficiently after reorganization."""

        async def test_async_embedding_integration():
            """Test async embedding integration."""
            from src.retrieval.embeddings import AsyncEmbeddingModel

            # Mock async embedding operations
            with patch.object(
                AsyncEmbeddingModel, "aembed_documents", new_callable=AsyncMock
            ) as mock_embed:
                mock_embed.return_value = [[0.1] * 1024 for _ in range(3)]

                model = AsyncEmbeddingModel()
                embeddings = await model.aembed_documents(["test1", "test2", "test3"])

                assert len(embeddings) == 3
                return embeddings

        # Benchmark async integration
        performance_tracker.benchmark_operation(
            "async_integration", test_async_embedding_integration
        )

        async_result = performance_tracker.measurements["async_integration"]

        # Async integration should be efficient
        assert async_result["duration_ms"] <= 50, (
            f"Async integration too slow: {async_result['duration_ms']:.1f}ms"
        )

        print(f"Async integration: {async_result['duration_ms']:.1f}ms")


@pytest.mark.performance
class TestStructuralPerformanceRegression:
    """Test for performance regressions after structural changes."""

    def test_overall_performance_validation(self, performance_tracker):
        """Comprehensive performance validation against all targets."""

        # Run a comprehensive test that exercises multiple components
        def comprehensive_workflow():
            """Simulate a typical DocMind AI workflow."""
            # Load configuration
            settings = DocMindSettings()

            # Get various configurations
            vllm_config = settings.get_vllm_config()
            agent_config = settings.get_agent_config()
            embedding_config = settings.get_embedding_config()

            # Import key modules
            from src.agents.coordinator import AgentCoordinator
            from src.models.processing import ProcessingConfig
            from src.retrieval.embeddings import EmbeddingModel

            # Create instances
            processing_config = ProcessingConfig()
            embedding_model = EmbeddingModel()
            coordinator = AgentCoordinator()

            return {
                "settings": settings,
                "configs": {
                    "vllm": vllm_config,
                    "agent": agent_config,
                    "embedding": embedding_config,
                },
                "instances": {
                    "processing": processing_config,
                    "embedding": embedding_model,
                    "coordinator": coordinator,
                },
            }

        # Benchmark comprehensive workflow
        performance_tracker.benchmark_operation(
            "comprehensive_workflow", comprehensive_workflow
        )

        workflow_result = performance_tracker.measurements["comprehensive_workflow"]

        # Comprehensive workflow should complete efficiently
        assert workflow_result["success"], (
            f"Comprehensive workflow failed: {workflow_result.get('error')}"
        )

        assert workflow_result["duration_ms"] <= 1000, (
            f"Comprehensive workflow too slow: {workflow_result['duration_ms']:.1f}ms"
        )

        assert workflow_result["rss_delta_mb"] <= 100, (
            f"Comprehensive workflow uses too much memory: "
            f"{workflow_result['rss_delta_mb']:.1f}MB"
        )

        # Validate all performance targets
        validation_results = performance_tracker.validate_performance_targets(
            IMPORT_PERFORMANCE_TARGETS
        )

        print(
            f"Comprehensive workflow: {workflow_result['duration_ms']:.1f}ms, "
            f"{workflow_result['rss_delta_mb']:.1f}MB"
        )
        print(f"Performance validation: {validation_results}")

        # All critical targets should be met
        critical_targets = ["config_loading_ms", "single_module_import_ms"]
        failed_critical = [
            t
            for t in validation_results["failed_targets"]
            if any(c in t for c in critical_targets)
        ]

        assert len(failed_critical) == 0, (
            f"Critical performance targets failed: {failed_critical}\n"
            f"Results: {validation_results['results']}"
        )

        # Overall should meet most targets (allow some tolerance)
        success_rate = 1 - (
            len(validation_results["failed_targets"]) / len(IMPORT_PERFORMANCE_TARGETS)
        )
        assert success_rate >= 0.8, (
            f"Too many performance targets failed: {success_rate:.1%} success rate"
        )


# Mock implementations for testing
class AsyncMock:
    """Simple async mock for testing."""

    def __init__(self, return_value=None):
        """Initialize async mock.

        Args:
            return_value: Value to return when called.
        """
        self.return_value = return_value

    async def __call__(self, *args, **kwargs):
        """Async callable implementation.

        Args:
            *args: Positional arguments (unused).
            **kwargs: Keyword arguments (unused).

        Returns:
            The configured return value.
        """
        return self.return_value


# Module mock classes
class EmbeddingModel:
    """Mock embedding model for testing."""

    pass


class AsyncEmbeddingModel:
    """Mock async embedding model for testing."""

    async def aembed_documents(self, texts):
        """Mock document embedding method.

        Args:
            texts: List of text documents to embed.

        Returns:
            List of mock embeddings (1024-dimensional vectors).
        """
        return [[0.1] * 1024 for _ in texts]


class DocumentLoader:
    """Mock document loader for testing."""

    pass


class DocumentProcessor:
    """Mock document processor for testing."""

    pass


class HybridPersistence:
    """Mock hybrid persistence for testing."""

    pass


class AgentCoordinator:
    """Mock agent coordinator for testing."""

    pass


def get_available_tools():
    """Mock function to get available tools."""
    return ["tool1", "tool2", "tool3"]


# Helper functions for patching
def mock_get_available_tools():
    """Mock implementation of get_available_tools."""
    return ["mock_tool_1", "mock_tool_2"]
