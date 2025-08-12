"""Integration tests for DocMind AI infrastructure components.

This module provides comprehensive integration testing for infrastructure components,
testing their interactions and ensuring proper cross-module functionality.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from src.core.infrastructure.gpu_monitor import gpu_performance_monitor
from src.core.infrastructure.hardware_utils import (
    detect_hardware,
    get_optimal_providers,
    get_recommended_batch_size,
)
from src.core.infrastructure.spacy_manager import SpacyManager, get_spacy_manager


class TestInfrastructureIntegration:
    """Test integration between infrastructure components."""

    @pytest.mark.integration
    def test_hardware_detection_gpu_monitor_consistency(self):
        """Test consistency between hardware detection and GPU monitoring."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=2684354560),
            patch("torch.cuda.memory_reserved", return_value=3221225472),
            patch("os.cpu_count", return_value=8),
        ):
            mock_device = Mock()
            mock_device.name = "NVIDIA RTX 3080"
            mock_device.total_memory = 10737418240  # 10 GB
            mock_device.major = 8
            mock_device.minor = 6
            mock_props.return_value = mock_device

            # Test hardware detection
            hardware_info = detect_hardware()

            assert hardware_info["cuda_available"] is True
            assert hardware_info["gpu_name"] == "NVIDIA RTX 3080"
            assert hardware_info["vram_total_gb"] == 10.0

            # Test GPU monitoring provides consistent data
            async def test_monitor():
                async with gpu_performance_monitor() as metrics:
                    assert metrics is not None
                    assert metrics.device_name == hardware_info["gpu_name"]
                    return metrics

            # Run the async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                gpu_metrics = loop.run_until_complete(test_monitor())
                assert gpu_metrics.device_name == hardware_info["gpu_name"]
            finally:
                loop.close()

    @pytest.mark.integration
    def test_hardware_providers_batch_size_consistency(self):
        """Test provider selection and batch size recommendations consistency."""
        # Test high-end GPU scenario
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties") as mock_props,
        ):
            mock_device = Mock()
            mock_device.total_memory = 21474836480  # 20 GB
            mock_props.return_value = mock_device

            providers = get_optimal_providers()
            batch_size = get_recommended_batch_size("embedding")

            assert "CUDAExecutionProvider" in providers
            assert batch_size == 128  # High-end GPU batch size

        # Test CPU-only scenario
        with patch("torch.cuda.is_available", return_value=False):
            providers = get_optimal_providers()
            batch_size = get_recommended_batch_size("embedding")

            assert providers == ["CPUExecutionProvider"]
            assert batch_size == 16  # CPU-only batch size

        # Test forced CPU scenario
        with patch("torch.cuda.is_available", return_value=True):
            providers = get_optimal_providers(force_cpu=True)
            # Batch size should still be based on actual hardware
            batch_size_gpu = get_recommended_batch_size("embedding")

            assert providers == ["CPUExecutionProvider"]
            # But batch size reflects actual GPU capability
            assert batch_size_gpu > 16  # Should be GPU-optimized

    @pytest.mark.integration
    def test_spacy_manager_hardware_integration(self):
        """Test spaCy manager integration with hardware detection."""
        detect_hardware()
        manager = get_spacy_manager()

        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
            patch("src.core.infrastructure.spacy_manager.download"),
        ):
            mock_nlp = Mock()
            mock_nlp.memory_zone.return_value.__enter__ = Mock(return_value=mock_nlp)
            mock_nlp.memory_zone.return_value.__exit__ = Mock(return_value=None)
            mock_load.return_value = mock_nlp

            # Load model using hardware-informed batch size
            batch_size = get_recommended_batch_size("llm")  # spaCy is like a small LLM
            nlp = manager.ensure_model("en_core_web_sm")

            assert nlp is mock_nlp
            assert isinstance(batch_size, int)
            assert batch_size >= 1

            # Test memory optimization works regardless of hardware
            with manager.memory_optimized_processing(
                "en_core_web_sm"
            ) as processing_nlp:
                assert processing_nlp is mock_nlp

    @pytest.mark.integration
    async def test_async_infrastructure_coordination(self):
        """Test async coordination between GPU monitoring and other components."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=1073741824),
            patch("torch.cuda.memory_reserved", return_value=2147483648),
            patch("os.cpu_count", return_value=16),
        ):
            mock_device = Mock()
            mock_device.name = "Multi-GPU System"
            mock_device.total_memory = 17179869184  # 16 GB
            mock_device.major = 8
            mock_device.minor = 0
            mock_props.return_value = mock_device

            # Start GPU monitoring
            async with gpu_performance_monitor() as gpu_metrics:
                assert gpu_metrics is not None

                # Get hardware info while monitoring
                hardware_info = detect_hardware()

                # Get optimal configuration
                providers = get_optimal_providers()
                batch_sizes = {
                    model_type: get_recommended_batch_size(model_type)
                    for model_type in ["embedding", "llm", "vision"]
                }

                # Verify coordination
                assert gpu_metrics.device_name == hardware_info["gpu_name"]
                assert hardware_info["gpu_device_count"] == 2
                assert "CUDAExecutionProvider" in providers
                assert all(batch_size > 1 for batch_size in batch_sizes.values())

    @pytest.mark.integration
    def test_error_propagation_across_components(self):
        """Test error handling coordination across infrastructure components."""
        # Test CUDA error propagation
        with patch("torch.cuda.is_available", side_effect=RuntimeError("CUDA failed")):
            # Hardware detection should fail
            with pytest.raises(RuntimeError, match="CUDA failed"):
                detect_hardware()

            # Provider selection should fail
            with pytest.raises(RuntimeError, match="CUDA failed"):
                get_optimal_providers()

            # Batch size recommendation should fail
            with pytest.raises(RuntimeError, match="CUDA failed"):
                get_recommended_batch_size("embedding")

        # Test spaCy error handling independent of hardware
        with (
            patch("torch.cuda.is_available", return_value=False),  # No GPU
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=False
            ),
            patch(
                "src.core.infrastructure.spacy_manager.download",
                side_effect=Exception("Download failed"),
            ),
        ):
            hardware_info = detect_hardware()
            assert hardware_info["cuda_available"] is False

            manager = SpacyManager()
            with pytest.raises(Exception, match="Download failed"):
                manager.ensure_model("test_model")

    @pytest.mark.integration
    @pytest.mark.performance
    def test_infrastructure_performance_coordination(self):
        """Test performance characteristics of coordinated infrastructure usage."""
        start_time = time.time()

        # Simulate typical startup sequence
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=0),
            patch("torch.cuda.memory_reserved", return_value=0),
            patch("os.cpu_count", return_value=8),
        ):
            mock_device = Mock()
            mock_device.name = "Performance GPU"
            mock_device.total_memory = 8589934592  # 8 GB
            mock_device.major = 7
            mock_device.minor = 5
            mock_props.return_value = mock_device

            # Detect hardware
            hardware_info = detect_hardware()
            detection_time = time.time() - start_time

            # Get providers and batch sizes
            providers = get_optimal_providers()
            batch_sizes = {
                model_type: get_recommended_batch_size(model_type)
                for model_type in ["embedding", "llm", "vision"]
            }
            config_time = time.time() - start_time - detection_time

            # Initialize spaCy manager
            with (
                patch(
                    "src.core.infrastructure.spacy_manager.is_package",
                    return_value=True,
                ),
                patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
                patch("src.core.infrastructure.spacy_manager.download"),
            ):
                mock_nlp = Mock()
                mock_load.return_value = mock_nlp

                manager = get_spacy_manager()
                nlp = manager.ensure_model("en_core_web_sm")
                spacy_time = time.time() - start_time - detection_time - config_time

        total_time = time.time() - start_time

        # Performance assertions
        assert detection_time < 0.01  # Hardware detection should be very fast
        assert config_time < 0.01  # Configuration should be very fast
        assert spacy_time < 0.05  # spaCy setup should be reasonable
        assert total_time < 0.1  # Total setup should be under 100ms

        # Verify results
        assert hardware_info["cuda_available"] is True
        assert "CUDAExecutionProvider" in providers
        assert all(isinstance(batch_size, int) for batch_size in batch_sizes.values())
        assert nlp is mock_nlp

    @pytest.mark.integration
    def test_configuration_validation_integration(self):
        """Test that infrastructure components provide valid configurations."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=1073741824),
            patch("torch.cuda.memory_reserved", return_value=1073741824),
            patch("os.cpu_count", return_value=12),
        ):
            mock_device = Mock()
            mock_device.name = "Validation GPU"
            mock_device.total_memory = 12884901888  # 12 GB
            mock_device.major = 8
            mock_device.minor = 6
            mock_props.return_value = mock_device

            # Get all configurations
            hardware_info = detect_hardware()
            providers = get_optimal_providers()
            batch_sizes = {
                model_type: get_recommended_batch_size(model_type)
                for model_type in ["embedding", "llm", "vision", "unknown"]
            }

            # Validate hardware info structure
            required_hardware_keys = {
                "cuda_available",
                "gpu_name",
                "vram_total_gb",
                "vram_available_gb",
                "gpu_compute_capability",
                "gpu_device_count",
                "fastembed_providers",
                "cpu_cores",
                "cpu_threads",
            }
            assert set(hardware_info.keys()) == required_hardware_keys
            assert hardware_info["cuda_available"] is True
            assert hardware_info["gpu_device_count"] == 1
            assert hardware_info["cpu_cores"] == 12

            # Validate providers
            assert isinstance(providers, list)
            assert len(providers) >= 1
            assert "CUDAExecutionProvider" in providers
            assert "CPUExecutionProvider" in providers

            # Validate batch sizes
            assert all(
                isinstance(batch_size, int) for batch_size in batch_sizes.values()
            )
            assert all(batch_size >= 1 for batch_size in batch_sizes.values())
            assert all(batch_size <= 128 for batch_size in batch_sizes.values())

            # Validate relationships
            assert hardware_info["vram_total_gb"] == 12.0
            assert hardware_info["vram_available_gb"] == 11.0  # 12 - 1 allocated
            assert (
                batch_sizes["embedding"] >= batch_sizes["llm"]
            )  # Embeddings can use larger batches

    @pytest.mark.integration
    async def test_real_world_workflow_simulation(self):
        """Simulate a real-world DocMind AI startup workflow."""
        workflow_results = {
            "hardware_detected": False,
            "gpu_monitored": False,
            "spacy_loaded": False,
            "configuration_optimized": False,
            "errors": [],
        }

        try:
            # Step 1: Detect hardware
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.device_count", return_value=1),
                patch("torch.cuda.get_device_properties") as mock_props,
                patch("torch.cuda.memory_allocated", return_value=0),
                patch("torch.cuda.memory_reserved", return_value=0),
                patch("os.cpu_count", return_value=8),
            ):
                mock_device = Mock()
                mock_device.name = "Workflow GPU"
                mock_device.total_memory = 10737418240  # 10 GB
                mock_device.major = 8
                mock_device.minor = 6
                mock_props.return_value = mock_device

                detect_hardware()
                workflow_results["hardware_detected"] = True

                # Step 2: Start GPU monitoring
                async with gpu_performance_monitor() as gpu_metrics:
                    if gpu_metrics is not None:
                        workflow_results["gpu_monitored"] = True

                        # Step 3: Configure optimal settings
                        get_optimal_providers()
                        {
                            "embedding": get_recommended_batch_size("embedding"),
                            "llm": get_recommended_batch_size("llm"),
                        }
                        workflow_results["configuration_optimized"] = True

                        # Step 4: Initialize spaCy manager
                        with (
                            patch(
                                "src.core.infrastructure.spacy_manager.is_package",
                                return_value=True,
                            ),
                            patch(
                                "src.core.infrastructure.spacy_manager.spacy.load"
                            ) as mock_load,
                            patch(
                                "src.core.infrastructure.spacy_manager.subprocess.run"
                            ),
                        ):
                            mock_nlp = Mock()
                            mock_nlp.memory_zone.return_value.__enter__ = Mock(
                                return_value=mock_nlp
                            )
                            mock_nlp.memory_zone.return_value.__exit__ = Mock(
                                return_value=None
                            )
                            mock_load.return_value = mock_nlp

                            manager = get_spacy_manager()
                            manager.ensure_model("en_core_web_sm")

                            # Test memory-optimized processing
                            with manager.memory_optimized_processing(
                                "en_core_web_sm"
                            ) as processing_nlp:
                                assert processing_nlp is mock_nlp
                                workflow_results["spacy_loaded"] = True

        except Exception as e:
            workflow_results["errors"].append(str(e))

        # Verify workflow success
        assert workflow_results["hardware_detected"]
        assert workflow_results["gpu_monitored"]
        assert workflow_results["configuration_optimized"]
        assert workflow_results["spacy_loaded"]
        assert len(workflow_results["errors"]) == 0


class TestInfrastructureStressTests:
    """Stress tests for infrastructure components under load."""

    @pytest.mark.integration
    @pytest.mark.performance
    def test_concurrent_hardware_detection(self):
        """Test concurrent hardware detection calls."""
        import threading

        results = []
        errors = []

        def detect_hardware_thread():
            try:
                with (
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.cuda.device_count", return_value=1),
                    patch("torch.cuda.get_device_properties") as mock_props,
                    patch("torch.cuda.memory_allocated", return_value=0),
                    patch("os.cpu_count", return_value=4),
                ):
                    mock_device = Mock()
                    mock_device.name = "Concurrent GPU"
                    mock_device.total_memory = 8589934592
                    mock_device.major = 7
                    mock_device.minor = 5
                    mock_props.return_value = mock_device

                    result = detect_hardware()
                    results.append(result)
            except Exception as e:
                errors.append(e)

        # Run 20 concurrent hardware detections
        threads = [threading.Thread(target=detect_hardware_thread) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All should succeed with identical results
        assert len(errors) == 0
        assert len(results) == 20
        assert all(result == results[0] for result in results)

    @pytest.mark.integration
    @pytest.mark.performance
    async def test_rapid_gpu_monitoring_cycles(self):
        """Test rapid GPU monitoring cycles."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=1073741824),
            patch("torch.cuda.memory_reserved", return_value=2147483648),
        ):
            mock_device = Mock()
            mock_device.name = "Stress Test GPU"
            mock_device.total_memory = 8589934592
            mock_props.return_value = mock_device

            start_time = time.time()

            # Run 50 rapid monitoring cycles
            for i in range(50):
                async with gpu_performance_monitor() as metrics:
                    assert metrics is not None
                    assert metrics.device_name == "Stress Test GPU"
                    assert metrics.memory_allocated_gb == pytest.approx(1.0, rel=1e-1)

                    # Brief processing simulation
                    await asyncio.sleep(0.001)  # 1ms

            elapsed_time = time.time() - start_time

            # Should complete 50 cycles in under 1 second
            assert elapsed_time < 1.0

    @pytest.mark.integration
    @pytest.mark.performance
    def test_spacy_manager_load_testing(self):
        """Test spaCy manager under concurrent load."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
            patch("src.core.infrastructure.spacy_manager.download"),
        ):
            mock_nlp = Mock()
            mock_nlp.memory_zone.return_value.__enter__ = Mock(return_value=mock_nlp)
            mock_nlp.memory_zone.return_value.__exit__ = Mock(return_value=None)
            mock_load.return_value = mock_nlp

            manager = get_spacy_manager()

            import threading

            results = []
            errors = []

            def load_and_process():
                try:
                    # Load model
                    nlp = manager.ensure_model("en_core_web_sm")
                    assert nlp is mock_nlp

                    # Use memory-optimized processing
                    with manager.memory_optimized_processing(
                        "en_core_web_sm"
                    ) as processing_nlp:
                        assert processing_nlp is mock_nlp
                        results.append(True)
                except Exception as e:
                    errors.append(e)

            # Run 30 concurrent load and process operations
            threads = [threading.Thread(target=load_and_process) for _ in range(30)]
            start_time = time.time()

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            elapsed_time = time.time() - start_time

            # All should succeed
            assert len(errors) == 0
            assert len(results) == 30
            assert all(result is True for result in results)

            # Should complete in reasonable time (under 5 seconds)
            assert elapsed_time < 5.0

            # spacy.load should only be called once due to caching
            assert mock_load.call_count == 1

    @pytest.mark.integration
    def test_memory_efficiency_coordination(self):
        """Test memory efficiency across infrastructure components."""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=0),
            patch("torch.cuda.memory_reserved", return_value=0),
            patch("os.cpu_count", return_value=8),
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
            patch("src.core.infrastructure.spacy_manager.download"),
        ):
            mock_device = Mock()
            mock_device.name = "Memory Test GPU"
            mock_device.total_memory = 8589934592
            mock_device.major = 8
            mock_device.minor = 0
            mock_props.return_value = mock_device

            mock_nlp = Mock()
            mock_nlp.memory_zone.return_value.__enter__ = Mock(return_value=mock_nlp)
            mock_nlp.memory_zone.return_value.__exit__ = Mock(return_value=None)
            mock_load.return_value = mock_nlp

            # Perform multiple operations
            for _ in range(10):
                # Hardware detection
                detect_hardware()

                # Provider configuration
                get_optimal_providers()
                {
                    model_type: get_recommended_batch_size(model_type)
                    for model_type in ["embedding", "llm", "vision"]
                }

                # spaCy operations
                manager = get_spacy_manager()
                manager.ensure_model("en_core_web_sm")
                with manager.memory_optimized_processing("en_core_web_sm"):
                    pass

                # Force garbage collection
                gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal (under 50MB for mocked operations)
        assert memory_increase < 50, f"Memory increased by {memory_increase:.2f}MB"
