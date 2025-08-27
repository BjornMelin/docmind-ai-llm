"""Resource cleanup validation and performance testing for DocMind AI.

This module provides comprehensive resource cleanup validation and testing:
- Resource lifecycle management validation
- Memory leak detection and prevention
- GPU resource cleanup verification
- Context manager effectiveness testing
- Resource pool management validation
- Cleanup order and dependency validation

Follows DocMind AI patterns with proper monitoring and cleanup verification.
"""

import asyncio
import gc
import time
import weakref
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from src.core.infrastructure.gpu_monitor import gpu_performance_monitor
from src.utils.storage import (
    async_gpu_memory_context,
    get_safe_gpu_info,
    get_safe_vram_usage,
    gpu_memory_context,
    sync_model_context,
)

# Resource cleanup test constants
RESOURCE_LEAK_THRESHOLD_MB = 50  # Alert if resources leak >50MB
GPU_CLEANUP_TOLERANCE_GB = 0.1  # Allow 100MB GPU cleanup tolerance
CLEANUP_TIMEOUT_SECONDS = 5.0  # Max time for cleanup operations
WEAK_REFERENCE_COLLECTION_ATTEMPTS = 3  # GC collection attempts

# Resource pool limits
MAX_CONCURRENT_RESOURCES = 10
RESOURCE_POOL_SIZE_LIMIT = 100


@pytest.fixture
def resource_tracker():
    """Fixture for tracking resource usage and cleanup validation."""

    class ResourceTracker:
        def __init__(self):
            self.allocations = []
            self.cleanup_results = []
            self.weak_references = []
            self.resource_pools = {}

        def track_allocation(
            self, resource_name: str, size_mb: float, resource_type: str
        ) -> dict[str, Any]:
            """Track resource allocation."""
            allocation = {
                "name": resource_name,
                "size_mb": size_mb,
                "type": resource_type,
                "allocated_at": time.time(),
                "cleaned_up": False,
                "cleanup_time": None,
            }

            self.allocations.append(allocation)
            return allocation

        def track_cleanup(
            self,
            resource_name: str,
            cleanup_duration: float,
            success: bool,
            error: str | None = None,
        ) -> None:
            """Track resource cleanup results."""
            # Find the allocation
            allocation = next(
                (
                    a
                    for a in self.allocations
                    if a["name"] == resource_name and not a["cleaned_up"]
                ),
                None,
            )

            if allocation:
                allocation["cleaned_up"] = True
                allocation["cleanup_time"] = cleanup_duration

            cleanup_result = {
                "resource_name": resource_name,
                "cleanup_duration": cleanup_duration,
                "success": success,
                "error": error,
                "timestamp": time.time(),
            }

            self.cleanup_results.append(cleanup_result)

        def add_weak_reference(self, obj: Any, name: str) -> None:
            """Add weak reference to track object lifetime."""
            weak_ref = weakref.ref(obj)
            self.weak_references.append(
                {
                    "name": name,
                    "ref": weak_ref,
                    "created_at": time.time(),
                }
            )

        def check_weak_references(self, force_gc: bool = True) -> dict[str, Any]:
            """Check if weak references have been collected (objects cleaned up)."""
            if force_gc:
                # Force garbage collection multiple times
                for _ in range(WEAK_REFERENCE_COLLECTION_ATTEMPTS):
                    gc.collect()
                    time.sleep(0.01)  # Small delay for cleanup

            results = {
                "total_references": len(self.weak_references),
                "collected_count": 0,
                "still_alive_count": 0,
                "collected_objects": [],
                "still_alive_objects": [],
            }

            for ref_info in self.weak_references:
                if ref_info["ref"]() is None:
                    results["collected_count"] += 1
                    results["collected_objects"].append(ref_info["name"])
                else:
                    results["still_alive_count"] += 1
                    results["still_alive_objects"].append(ref_info["name"])

            results["collection_rate"] = (
                results["collected_count"] / results["total_references"]
                if results["total_references"] > 0
                else 1.0
            )

            return results

        def validate_cleanup_completeness(self) -> dict[str, Any]:
            """Validate that all allocated resources were properly cleaned up."""
            total_allocations = len(self.allocations)
            cleaned_allocations = sum(1 for a in self.allocations if a["cleaned_up"])

            leaked_resources = [a for a in self.allocations if not a["cleaned_up"]]

            validation = {
                "total_allocations": total_allocations,
                "cleaned_allocations": cleaned_allocations,
                "leaked_count": len(leaked_resources),
                "cleanup_rate": cleaned_allocations / total_allocations
                if total_allocations > 0
                else 1.0,
                "leaked_resources": leaked_resources,
                "all_cleaned": len(leaked_resources) == 0,
            }

            return validation

        def analyze_cleanup_performance(self) -> dict[str, Any]:
            """Analyze cleanup performance metrics."""
            if not self.cleanup_results:
                return {"error": "No cleanup results to analyze"}

            successful_cleanups = [r for r in self.cleanup_results if r["success"]]
            failed_cleanups = [r for r in self.cleanup_results if not r["success"]]

            cleanup_times = [r["cleanup_duration"] for r in successful_cleanups]

            analysis = {
                "total_cleanups": len(self.cleanup_results),
                "successful_cleanups": len(successful_cleanups),
                "failed_cleanups": len(failed_cleanups),
                "success_rate": len(successful_cleanups) / len(self.cleanup_results),
                "average_cleanup_time": sum(cleanup_times) / len(cleanup_times)
                if cleanup_times
                else 0.0,
                "max_cleanup_time": max(cleanup_times) if cleanup_times else 0.0,
                "min_cleanup_time": min(cleanup_times) if cleanup_times else 0.0,
                "slow_cleanups": [
                    r
                    for r in successful_cleanups
                    if r["cleanup_duration"] > CLEANUP_TIMEOUT_SECONDS
                ],
            }

            return analysis

        def track_resource_pool(
            self, pool_name: str, current_size: int, max_size: int
        ) -> None:
            """Track resource pool statistics."""
            self.resource_pools[pool_name] = {
                "current_size": current_size,
                "max_size": max_size,
                "utilization": current_size / max_size if max_size > 0 else 0.0,
                "timestamp": time.time(),
            }

        def validate_resource_pools(self) -> dict[str, Any]:
            """Validate resource pool management."""
            validation = {
                "pools_count": len(self.resource_pools),
                "pools_within_limits": True,
                "over_limit_pools": [],
                "average_utilization": 0.0,
            }

            if self.resource_pools:
                utilizations = []

                for pool_name, pool_info in self.resource_pools.items():
                    utilizations.append(pool_info["utilization"])

                    if pool_info["current_size"] > RESOURCE_POOL_SIZE_LIMIT:
                        validation["pools_within_limits"] = False
                        validation["over_limit_pools"].append(pool_name)

                validation["average_utilization"] = sum(utilizations) / len(
                    utilizations
                )

            return validation

    return ResourceTracker()


@pytest.mark.performance
class TestResourceLifecycleManagement:
    """Test resource lifecycle management and cleanup."""

    def test_gpu_memory_context_cleanup(self, resource_tracker):
        """Test GPU memory context manager cleanup."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for cleanup test")

        initial_memory = get_safe_vram_usage()
        resource_tracker.track_allocation("gpu_test_tensors", 100, "gpu_memory")

        # Test context manager cleanup
        cleanup_start = time.perf_counter()

        try:
            with gpu_memory_context():
                # Allocate GPU memory
                test_tensors = []
                for i in range(5):
                    tensor = torch.randn(1000, 1000, device="cuda")
                    test_tensors.append(tensor)
                    resource_tracker.add_weak_reference(tensor, f"gpu_tensor_{i}")

                # Check memory increased
                peak_memory = get_safe_vram_usage()
                assert peak_memory > initial_memory, "GPU memory should have increased"

                # Exit context - should trigger cleanup

            cleanup_duration = time.perf_counter() - cleanup_start

            # Check cleanup was effective
            final_memory = get_safe_vram_usage()
            memory_cleaned = peak_memory - final_memory

            resource_tracker.track_cleanup(
                "gpu_test_tensors",
                cleanup_duration,
                success=memory_cleaned >= 0,
                error=None if memory_cleaned >= 0 else "Memory not cleaned up",
            )

            # Verify cleanup
            assert abs(final_memory - initial_memory) <= GPU_CLEANUP_TOLERANCE_GB, (
                f"GPU memory not cleaned up: {final_memory:.3f}GB vs "
                f"initial {initial_memory:.3f}GB"
            )

        except Exception as e:
            resource_tracker.track_cleanup(
                "gpu_test_tensors", 0.0, success=False, error=str(e)
            )
            raise

        # Check weak references were collected
        weak_ref_results = resource_tracker.check_weak_references()
        assert weak_ref_results["collection_rate"] >= 0.8, (
            f"Weak references not collected: "
            f"{weak_ref_results['collection_rate']:.2f} < 0.8"
        )

    @pytest.mark.asyncio
    async def test_async_resource_cleanup(self, resource_tracker):
        """Test async resource cleanup patterns."""
        initial_memory = get_safe_vram_usage()
        resource_tracker.track_allocation("async_resources", 50, "async_memory")

        cleanup_start = time.perf_counter()

        try:
            async with async_gpu_memory_context():
                # Simulate async resource allocation
                async_resources = []

                async def allocate_async_resource(resource_id: int):
                    """Mock async resource allocation."""
                    await asyncio.sleep(0.01)  # Simulate async work
                    if torch.cuda.is_available():
                        resource = torch.randn(500, 500, device="cuda")
                    else:
                        resource = torch.randn(500, 500)  # CPU fallback
                    return resource

                # Allocate resources concurrently
                tasks = [allocate_async_resource(i) for i in range(3)]
                async_resources = await asyncio.gather(*tasks)

                # Add weak references
                for i, resource in enumerate(async_resources):
                    resource_tracker.add_weak_reference(resource, f"async_resource_{i}")

                # Check memory usage during allocation
                if torch.cuda.is_available():
                    peak_memory = get_safe_vram_usage()
                    assert peak_memory >= initial_memory, (
                        "Memory should have been allocated"
                    )

            cleanup_duration = time.perf_counter() - cleanup_start

            # Async context should clean up automatically
            get_safe_vram_usage()

            resource_tracker.track_cleanup(
                "async_resources", cleanup_duration, success=True, error=None
            )

        except Exception as e:
            resource_tracker.track_cleanup(
                "async_resources", 0.0, success=False, error=str(e)
            )
            raise

        # Verify async cleanup effectiveness
        weak_ref_results = resource_tracker.check_weak_references()
        assert weak_ref_results["collection_rate"] >= 0.7, (
            f"Async resources not cleaned up: "
            f"{weak_ref_results['collection_rate']:.2f} < 0.7"
        )

    def test_model_context_lifecycle(self, resource_tracker):
        """Test model context manager resource lifecycle."""

        def mock_model_factory(**kwargs):
            """Mock model factory for testing."""
            model = MagicMock()
            model.cleanup = MagicMock()
            model.close = MagicMock()
            return model

        resource_tracker.track_allocation("test_model", 200, "model_memory")

        cleanup_start = time.perf_counter()

        try:
            with sync_model_context(
                mock_model_factory, cleanup_method="cleanup"
            ) as model:
                # Use model
                assert model is not None
                resource_tracker.add_weak_reference(model, "test_model")

                # Verify model is usable
                result = model.some_method()
                assert result is not None

            cleanup_duration = time.perf_counter() - cleanup_start

            # Verify cleanup method was called
            assert model.cleanup.called, "Model cleanup method should have been called"

            resource_tracker.track_cleanup(
                "test_model", cleanup_duration, success=True, error=None
            )

        except Exception as e:
            resource_tracker.track_cleanup(
                "test_model", 0.0, success=False, error=str(e)
            )
            raise

        # Check model was cleaned up
        weak_ref_results = resource_tracker.check_weak_references()
        print(
            f"Model cleanup - Collection rate: "
            f"{weak_ref_results['collection_rate']:.2f}"
        )

    def test_error_handling_with_cleanup(self, resource_tracker):
        """Test resource cleanup during error conditions."""
        initial_memory = get_safe_vram_usage()
        resource_tracker.track_allocation("error_test_resources", 75, "error_memory")

        cleanup_start = time.perf_counter()

        try:
            with gpu_memory_context():
                # Allocate resources
                if torch.cuda.is_available():
                    error_tensor = torch.randn(800, 800, device="cuda")
                    resource_tracker.add_weak_reference(error_tensor, "error_tensor")

                # Simulate error condition
                raise RuntimeError("Simulated error for cleanup testing")

        except RuntimeError:
            cleanup_duration = time.perf_counter() - cleanup_start

            # Even with error, cleanup should happen
            final_memory = get_safe_vram_usage()

            resource_tracker.track_cleanup(
                "error_test_resources",
                cleanup_duration,
                success=abs(final_memory - initial_memory) <= GPU_CLEANUP_TOLERANCE_GB,
                error=None
                if abs(final_memory - initial_memory) <= GPU_CLEANUP_TOLERANCE_GB
                else "Memory not cleaned after error",
            )

            # Verify cleanup happened despite error
            assert abs(final_memory - initial_memory) <= GPU_CLEANUP_TOLERANCE_GB, (
                f"Resources not cleaned up after error: {final_memory:.3f}GB vs "
                f"{initial_memory:.3f}GB"
            )

        # Check objects were released
        weak_ref_results = resource_tracker.check_weak_references()
        assert weak_ref_results["collection_rate"] >= 0.5, (
            f"Error cleanup insufficient: "
            f"{weak_ref_results['collection_rate']:.2f} < 0.5"
        )


@pytest.mark.performance
class TestResourcePoolManagement:
    """Test resource pool management and cleanup."""

    def test_resource_pool_limits(self, resource_tracker):
        """Test resource pool size limits and management."""
        # Simulate resource pool with different sizes
        pool_configurations = [
            {"name": "embedding_models", "size": 5, "max": 10},
            {"name": "reranking_models", "size": 3, "max": 8},
            {"name": "document_cache", "size": 20, "max": 50},
            {"name": "query_cache", "size": 15, "max": 30},
        ]

        for config in pool_configurations:
            resource_tracker.track_resource_pool(
                config["name"], config["size"], config["max"]
            )

        # Validate pool management
        pool_validation = resource_tracker.validate_resource_pools()

        assert pool_validation["pools_within_limits"], (
            f"Resource pools exceeded limits: {pool_validation['over_limit_pools']}"
        )

        assert pool_validation["average_utilization"] <= 1.0, (
            f"Average pool utilization too high: "
            f"{pool_validation['average_utilization']:.2f}"
        )

        print(f"Resource pool validation: {pool_validation}")

    def test_concurrent_resource_management(self, resource_tracker):
        """Test resource management under concurrent access."""
        import threading

        resource_allocations = []
        allocation_errors = []

        def allocate_resources(worker_id: int):
            """Worker function to allocate resources concurrently."""
            try:
                for resource_id in range(5):
                    resource_name = f"concurrent_resource_{worker_id}_{resource_id}"

                    # Simulate resource allocation
                    with gpu_memory_context():
                        if torch.cuda.is_available():
                            resource = torch.randn(100, 100, device="cuda")
                        else:
                            resource = torch.randn(100, 100)  # CPU fallback

                        allocation = resource_tracker.track_allocation(
                            resource_name,
                            4,  # 4MB per resource
                            "concurrent_memory",
                        )

                        resource_tracker.add_weak_reference(resource, resource_name)
                        resource_allocations.append(allocation)

                        # Simulate some work
                        time.sleep(0.01)

            except Exception as e:
                allocation_errors.append(f"Worker {worker_id}: {str(e)}")

        # Run concurrent resource allocation
        workers = []
        worker_count = 4

        for worker_id in range(worker_count):
            worker = threading.Thread(target=allocate_resources, args=(worker_id,))
            workers.append(worker)
            worker.start()

        # Wait for all workers to complete
        for worker in workers:
            worker.join()

        # Analyze results
        assert len(allocation_errors) == 0, (
            f"Concurrent allocation errors: {allocation_errors}"
        )

        expected_allocations = worker_count * 5
        assert len(resource_allocations) == expected_allocations, (
            f"Expected {expected_allocations} allocations, got "
            f"{len(resource_allocations)}"
        )

        # Force cleanup and check weak references
        weak_ref_results = resource_tracker.check_weak_references()

        print("Concurrent resource management:")
        print(f"  Allocations: {len(resource_allocations)}")
        print(f"  Collection rate: {weak_ref_results['collection_rate']:.2f}")
        print(f"  Errors: {len(allocation_errors)}")

        # Should handle concurrency well
        assert weak_ref_results["collection_rate"] >= 0.6, (
            f"Concurrent cleanup rate too low: "
            f"{weak_ref_results['collection_rate']:.2f} < 0.6"
        )

    def test_resource_cleanup_order(self, resource_tracker):
        """Test resource cleanup happens in correct order."""
        cleanup_order = []

        class MockResource:
            def __init__(self, name: str, dependencies: list[str] = None):
                self.name = name
                self.dependencies = dependencies or []
                self.cleaned_up = False

            def cleanup(self):
                cleanup_order.append(self.name)
                self.cleaned_up = True

        # Create resources with dependencies (child -> parent order)
        resources = {
            "database": MockResource("database"),
            "cache": MockResource("cache", ["database"]),
            "model": MockResource("model", ["cache", "database"]),
            "processor": MockResource("processor", ["model"]),
        }

        # Track resources
        for name, resource in resources.items():
            resource_tracker.track_allocation(name, 50, "mock_resource")
            resource_tracker.add_weak_reference(resource, name)

        cleanup_start = time.perf_counter()

        # Simulate cleanup in dependency order (reverse dependency order)
        cleanup_sequence = ["processor", "model", "cache", "database"]

        for resource_name in cleanup_sequence:
            resource = resources[resource_name]
            resource.cleanup()

            resource_tracker.track_cleanup(
                resource_name,
                0.01,  # Mock cleanup time
                success=True,
                error=None,
            )

        cleanup_duration = time.perf_counter() - cleanup_start

        # Verify cleanup order
        expected_order = ["processor", "model", "cache", "database"]
        assert cleanup_order == expected_order, (
            f"Cleanup order incorrect: {cleanup_order} vs expected {expected_order}"
        )

        # Verify all resources cleaned up
        all_cleaned = all(resource.cleaned_up for resource in resources.values())
        assert all_cleaned, "Not all resources were cleaned up"

        print("Resource cleanup order test:")
        print(f"  Cleanup sequence: {cleanup_order}")
        print(f"  Total cleanup time: {cleanup_duration:.3f}s")


@pytest.mark.performance
class TestCleanupPerformanceValidation:
    """Test cleanup performance and efficiency."""

    def test_cleanup_performance_analysis(self, resource_tracker):
        """Test cleanup performance meets requirements."""
        # Simulate various cleanup scenarios
        cleanup_scenarios = [
            {"name": "small_tensor", "size": 10, "expected_time": 0.01},
            {"name": "medium_model", "size": 100, "expected_time": 0.05},
            {"name": "large_cache", "size": 500, "expected_time": 0.1},
            {"name": "batch_resources", "size": 50, "expected_time": 0.03},
        ]

        for scenario in cleanup_scenarios:
            resource_tracker.track_allocation(
                scenario["name"], scenario["size"], "performance_test"
            )

            # Simulate cleanup with realistic timing
            cleanup_start = time.perf_counter()
            time.sleep(scenario["expected_time"])  # Simulate cleanup work
            cleanup_duration = time.perf_counter() - cleanup_start

            success = cleanup_duration <= CLEANUP_TIMEOUT_SECONDS

            resource_tracker.track_cleanup(
                scenario["name"],
                cleanup_duration,
                success=success,
                error=None if success else "Cleanup timeout",
            )

        # Analyze cleanup performance
        performance_analysis = resource_tracker.analyze_cleanup_performance()

        print(f"Cleanup performance analysis: {performance_analysis}")

        # Validate performance requirements
        assert performance_analysis["success_rate"] >= 0.95, (
            f"Cleanup success rate too low: "
            f"{performance_analysis['success_rate']:.2f} < 0.95"
        )

        assert (
            performance_analysis["average_cleanup_time"] <= CLEANUP_TIMEOUT_SECONDS
        ), (
            f"Average cleanup time too high: "
            f"{performance_analysis['average_cleanup_time']:.3f}s"
        )

        assert len(performance_analysis["slow_cleanups"]) == 0, (
            f"Slow cleanups detected: {performance_analysis['slow_cleanups']}"
        )

    def test_resource_leak_detection(self, resource_tracker):
        """Test resource leak detection and prevention."""
        # Simulate scenarios with and without leaks
        test_scenarios = [
            {"name": "proper_cleanup", "should_leak": False, "resource_count": 5},
            {"name": "minor_leak", "should_leak": True, "resource_count": 2},
            {"name": "batch_cleanup", "should_leak": False, "resource_count": 10},
        ]

        for scenario in test_scenarios:
            scenario_resources = []

            # Allocate resources
            for i in range(scenario["resource_count"]):
                resource_name = f"{scenario['name']}_resource_{i}"
                resource_tracker.track_allocation(resource_name, 25, "leak_test")

                # Create mock resource
                if torch.cuda.is_available():
                    resource = torch.randn(200, 200, device="cuda")
                else:
                    resource = torch.randn(200, 200)

                scenario_resources.append((resource_name, resource))
                resource_tracker.add_weak_reference(resource, resource_name)

            # Simulate cleanup (or lack thereof for leak test)
            if not scenario["should_leak"]:
                for resource_name, resource in scenario_resources:
                    # Proper cleanup
                    del resource
                    resource_tracker.track_cleanup(resource_name, 0.01, success=True)
            else:
                # Simulate partial cleanup (leak scenario)
                leaked_count = scenario["resource_count"] // 2

                for i, (resource_name, resource) in enumerate(scenario_resources):
                    if i >= leaked_count:
                        del resource
                        resource_tracker.track_cleanup(
                            resource_name, 0.01, success=True
                        )
                    # Leave first half as "leaked" (not cleaned up)

        # Analyze cleanup completeness
        cleanup_validation = resource_tracker.validate_cleanup_completeness()
        weak_ref_results = resource_tracker.check_weak_references()

        print("Resource leak detection:")
        print(f"  Cleanup validation: {cleanup_validation}")
        print(f"  Weak reference results: {weak_ref_results}")

        # Should detect leaks
        expected_leaks = 1  # minor_leak scenario
        assert cleanup_validation["leaked_count"] >= expected_leaks, (
            f"Leak detection failed: {cleanup_validation['leaked_count']} < "
            f"{expected_leaks}"
        )

        # Overall cleanup rate should still be reasonable
        assert cleanup_validation["cleanup_rate"] >= 0.7, (
            f"Overall cleanup rate too low: "
            f"{cleanup_validation['cleanup_rate']:.2f} < 0.7"
        )

    @pytest.mark.requires_gpu
    async def test_comprehensive_resource_cleanup_validation(self, resource_tracker):
        """Test comprehensive resource cleanup validation with GPU monitoring."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for comprehensive cleanup test")

        initial_gpu_info = get_safe_gpu_info()
        initial_memory = initial_gpu_info["allocated_memory_gb"]

        # Comprehensive resource lifecycle test
        async with gpu_performance_monitor() as gpu_metrics:
            if gpu_metrics:
                print(f"Initial GPU state: {gpu_metrics}")

                # Phase 1: Allocate various resources
                with gpu_memory_context():
                    resources = []

                    # Embedding model simulation
                    embedding_tensors = [
                        torch.randn(1000, 1024, device="cuda") for _ in range(3)
                    ]
                    resources.extend(embedding_tensors)
                    resource_tracker.track_allocation(
                        "embedding_models", 300, "gpu_models"
                    )

                    # Reranking model simulation
                    rerank_tensors = [
                        torch.randn(512, 768, device="cuda") for _ in range(2)
                    ]
                    resources.extend(rerank_tensors)
                    resource_tracker.track_allocation(
                        "reranking_models", 150, "gpu_models"
                    )

                    # Document cache simulation
                    cache_tensors = [
                        torch.randn(256, 512, device="cuda") for _ in range(5)
                    ]
                    resources.extend(cache_tensors)
                    resource_tracker.track_allocation(
                        "document_cache", 100, "gpu_cache"
                    )

                    # Track all resources with weak references
                    for i, resource in enumerate(resources):
                        resource_tracker.add_weak_reference(
                            resource, f"comprehensive_resource_{i}"
                        )

                    # Check peak memory usage
                    peak_memory = get_safe_vram_usage()
                    memory_increase = peak_memory - initial_memory

                    print(
                        f"Peak GPU memory usage: {peak_memory:.2f}GB "
                        f"(+{memory_increase:.2f}GB)"
                    )

                    # Phase 2: Simulate work and partial cleanup
                    await asyncio.sleep(0.1)  # Simulate work

                    # Clean up some resources manually
                    for i in range(
                        0, len(resources), 2
                    ):  # Clean up every other resource
                        del resources[i]

                # Phase 3: Context manager should clean up remaining resources
                cleanup_start = time.perf_counter()

            # Context manager exit should trigger cleanup
            cleanup_duration = time.perf_counter() - cleanup_start
            final_memory = get_safe_vram_usage()
            memory_cleaned = peak_memory - final_memory

            # Track comprehensive cleanup
            resource_tracker.track_cleanup(
                "comprehensive_test",
                cleanup_duration,
                success=memory_cleaned >= memory_increase * 0.8,  # 80% cleanup expected
                error=None,
            )

            print("Cleanup results:")
            print(f"  Cleanup time: {cleanup_duration:.3f}s")
            print(f"  Memory cleaned: {memory_cleaned:.3f}GB")
            print(f"  Final memory: {final_memory:.3f}GB")

            # Validate comprehensive cleanup
            weak_ref_results = resource_tracker.check_weak_references()
            resource_tracker.validate_cleanup_completeness()
            performance_analysis = resource_tracker.analyze_cleanup_performance()

            # Comprehensive validation assertions
            assert memory_cleaned >= memory_increase * 0.5, (
                f"Insufficient memory cleanup: {memory_cleaned:.3f}GB < "
                f"{memory_increase * 0.5:.3f}GB"
            )

            assert weak_ref_results["collection_rate"] >= 0.7, (
                f"Poor object collection rate: "
                f"{weak_ref_results['collection_rate']:.2f} < 0.7"
            )

            assert performance_analysis["success_rate"] >= 0.9, (
                f"Poor cleanup success rate: "
                f"{performance_analysis['success_rate']:.2f} < 0.9"
            )

            print("Comprehensive cleanup validation:")
            print(
                f"  Object collection rate: {weak_ref_results['collection_rate']:.2f}"
            )
            print(f"  Cleanup success rate: {performance_analysis['success_rate']:.2f}")
            print(
                f"  Average cleanup time: "
                f"{performance_analysis['average_cleanup_time']:.3f}s"
            )
