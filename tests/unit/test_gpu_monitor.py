"""Comprehensive tests for PyTorch native GPU monitoring.

This module provides comprehensive test coverage for GPU monitoring functionality,
including edge cases, error conditions, performance benchmarks, and memory management.
"""

import asyncio
import os
from unittest.mock import Mock, patch

import pytest

# Performance tests require explicit opt-in
pytestmark = pytest.mark.skipif(os.getenv("PYTEST_PERF") != "1", reason="perf disabled")

torch = pytest.importorskip("torch")

from src.core.infrastructure.gpu_monitor import GPUMetrics, gpu_performance_monitor


class TestGPUMetricsCore:
    """Test GPUMetrics dataclass."""

    def test_gpu_metrics_creation(self):
        """Test GPUMetrics creation with valid data."""
        gpu_metrics = GPUMetrics(
            device_name="NVIDIA GeForce RTX 3080",
            memory_allocated_gb=2.5,
            memory_reserved_gb=3.0,
            utilization_percent=80.0,
        )

        assert gpu_metrics.device_name == "NVIDIA GeForce RTX 3080"
        assert gpu_metrics.memory_allocated_gb == 2.5
        assert gpu_metrics.memory_reserved_gb == 3.0
        assert gpu_metrics.utilization_percent == 80.0

    def test_gpu_metrics_immutable(self):
        """Test that GPUMetrics is immutable (frozen=True)."""
        gpu_metrics = GPUMetrics("Test", 1.0, 2.0, 50.0)

        with pytest.raises(AttributeError):
            gpu_metrics.device_name = "Modified"


@pytest.mark.asyncio
class TestGPUPerformanceMonitor:
    """Test GPU performance monitoring context manager."""

    @patch("torch.cuda.is_available", return_value=False)
    async def test_no_cuda_returns_none(self, mock_cuda_available):
        """Test that context manager returns None when CUDA unavailable."""
        async with gpu_performance_monitor() as gpu_metrics:
            assert gpu_metrics is None

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", return_value=2684354560)  # 2.5 GB
    @patch("torch.cuda.memory_reserved", return_value=3221225472)  # 3.0 GB
    async def test_cuda_returns_metrics(
        self, mock_reserved, mock_allocated, mock_props, mock_available
    ):
        """Test that context manager returns GPUMetrics when CUDA available."""
        # Mock device properties
        mock_device = Mock()
        mock_device.name = "NVIDIA GeForce RTX 3080"
        mock_device.total_memory = 10737418240  # 10 GB
        mock_props.return_value = mock_device

        async with gpu_performance_monitor() as gpu_metrics:
            assert gpu_metrics is not None
            assert isinstance(gpu_metrics, GPUMetrics)
            assert gpu_metrics.device_name == "NVIDIA GeForce RTX 3080"
            assert gpu_metrics.memory_allocated_gb == pytest.approx(2.5, rel=1e-1)
            assert gpu_metrics.memory_reserved_gb == pytest.approx(3.0, rel=1e-1)
            assert gpu_metrics.utilization_percent == pytest.approx(25.0, rel=1e-1)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", return_value=10737418240)  # 10 GB (full)
    @patch("torch.cuda.memory_reserved", return_value=10737418240)  # 10 GB
    async def test_utilization_capped_at_100(
        self, mock_reserved, mock_allocated, mock_props, mock_available
    ):
        """Test that utilization is capped at 100%."""
        # Mock device properties
        mock_device = Mock()
        mock_device.name = "Test GPU"
        mock_device.total_memory = 10737418240  # 10 GB
        mock_props.return_value = mock_device

        async with gpu_performance_monitor() as gpu_metrics:
            assert gpu_metrics is not None
            assert gpu_metrics.utilization_percent == 100.0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", side_effect=RuntimeError("CUDA error"))
    @patch("torch.cuda.memory_reserved", return_value=0)
    async def test_cuda_memory_error_handling(
        self, mock_reserved, mock_allocated, mock_props, mock_available
    ):
        """Test handling of CUDA memory errors."""
        mock_device = Mock()
        mock_device.name = "Error GPU"
        mock_device.total_memory = 8589934592  # 8 GB
        mock_props.return_value = mock_device

        # Should raise the original exception
        with pytest.raises(RuntimeError, match="CUDA error"):
            async with gpu_performance_monitor():
                pass

    @patch("torch.cuda.is_available", return_value=True)
    @patch(
        "torch.cuda.get_device_properties", side_effect=RuntimeError("Device not found")
    )
    async def test_device_properties_error(self, mock_props, mock_available):
        """Test handling of device properties errors."""
        with pytest.raises(RuntimeError, match="Device not found"):
            async with gpu_performance_monitor():
                pass

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", return_value=0)
    @patch("torch.cuda.memory_reserved", return_value=0)
    async def test_zero_memory_usage(
        self, mock_reserved, mock_allocated, mock_props, mock_available
    ):
        """Test metrics with zero memory usage."""
        mock_device = Mock()
        mock_device.name = "Empty GPU"
        mock_device.total_memory = 8589934592  # 8 GB
        mock_props.return_value = mock_device

        async with gpu_performance_monitor() as gpu_metrics:
            assert gpu_metrics is not None
            assert gpu_metrics.memory_allocated_gb == 0.0
            assert gpu_metrics.memory_reserved_gb == 0.0
            assert gpu_metrics.utilization_percent == 0.0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", return_value=1073741824)  # 1 GB
    @patch("torch.cuda.memory_reserved", return_value=2147483648)  # 2 GB
    async def test_small_gpu_memory(
        self, mock_reserved, mock_allocated, mock_props, mock_available
    ):
        """Test metrics with small GPU memory configuration."""
        mock_device = Mock()
        mock_device.name = "GTX 1050"
        mock_device.total_memory = 2147483648  # 2 GB
        mock_props.return_value = mock_device

        async with gpu_performance_monitor() as gpu_metrics:
            assert gpu_metrics is not None
            assert gpu_metrics.device_name == "GTX 1050"
            assert gpu_metrics.memory_allocated_gb == pytest.approx(1.0, rel=1e-1)
            assert gpu_metrics.memory_reserved_gb == pytest.approx(2.0, rel=1e-1)
            assert gpu_metrics.utilization_percent == pytest.approx(50.0, rel=1e-1)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", return_value=85899345920)  # 80 GB
    @patch("torch.cuda.memory_reserved", return_value=85899345920)  # 80 GB
    async def test_large_gpu_memory(
        self, mock_reserved, mock_allocated, mock_props, mock_available
    ):
        """Test metrics with large GPU memory (A100/H100)."""
        mock_device = Mock()
        mock_device.name = "NVIDIA A100-SXM4-80GB"
        mock_device.total_memory = 85899345920  # 80 GB
        mock_props.return_value = mock_device

        async with gpu_performance_monitor() as gpu_metrics:
            assert gpu_metrics is not None
            assert "A100" in gpu_metrics.device_name
            assert gpu_metrics.memory_allocated_gb == pytest.approx(80.0, rel=1e-1)
            assert gpu_metrics.memory_reserved_gb == pytest.approx(80.0, rel=1e-1)
            assert gpu_metrics.utilization_percent == pytest.approx(100.0, rel=1e-1)

    async def test_concurrent_monitoring(self):
        """Test concurrent GPU monitoring contexts."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=2684354560),
            patch("torch.cuda.memory_reserved", return_value=3221225472),
        ):
            mock_device = Mock()
            mock_device.name = "Concurrent GPU"
            mock_device.total_memory = 10737418240
            mock_props.return_value = mock_device

            # Test multiple concurrent contexts
            async def monitor_gpu():
                async with gpu_performance_monitor() as gpu_metrics:
                    assert gpu_metrics is not None
                    return gpu_metrics

            # Run 5 concurrent monitoring contexts
            results = await asyncio.gather(*[monitor_gpu() for _ in range(5)])

            assert len(results) == 5
            for metrics in results:
                assert metrics.device_name == "Concurrent GPU"
                assert metrics.memory_allocated_gb == pytest.approx(2.5, rel=1e-1)

    @pytest.mark.performance
    async def test_monitoring_performance(self, benchmark_config):
        """Benchmark GPU monitoring performance."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=2684354560),
            patch("torch.cuda.memory_reserved", return_value=3221225472),
        ):
            mock_device = Mock()
            mock_device.name = "Performance GPU"
            mock_device.total_memory = 10737418240
            mock_props.return_value = mock_device

            # Measure time for 100 monitor cycles
            import time

            start_time = time.time()

            for _ in range(100):
                async with gpu_performance_monitor() as gpu_metrics:
                    assert gpu_metrics is not None

            elapsed_time = time.time() - start_time

            # Should complete 100 cycles in under 1 second
            assert elapsed_time < 1.0
            # Average time per cycle should be under 10ms
            assert (elapsed_time / 100) < 0.01


class TestGPUMetricsEdgeCases:
    """Test GPUMetrics edge cases and validation."""

    def test_gpu_metrics_with_extreme_values(self):
        """Test GPUMetrics with extreme values."""
        # Test with very large values
        large_gpu_metrics = GPUMetrics(
            device_name="High-End GPU",
            memory_allocated_gb=999.9,
            memory_reserved_gb=1000.0,
            utilization_percent=100.0,
        )
        assert large_gpu_metrics.memory_allocated_gb == 999.9
        assert large_gpu_metrics.utilization_percent == 100.0

        # Test with very small values
        small_gpu_metrics = GPUMetrics(
            device_name="Low-End GPU",
            memory_allocated_gb=0.001,
            memory_reserved_gb=0.002,
            utilization_percent=0.1,
        )
        assert small_gpu_metrics.memory_allocated_gb == 0.001
        assert small_gpu_metrics.utilization_percent == 0.1

    def test_gpu_metrics_string_representation(self):
        """Test string representation of GPUMetrics."""
        gpu_gpu_metrics = GPUMetrics(
            device_name="Test GPU",
            memory_allocated_gb=2.5,
            memory_reserved_gb=3.0,
            utilization_percent=83.33,
        )

        str_repr = str(gpu_gpu_metrics)
        assert "Test GPU" in str_repr
        assert "2.5" in str_repr
        assert "3.0" in str_repr
        assert "83.33" in str_repr

    def test_gpu_metrics_equality(self):
        """Test GPUMetrics equality comparison."""
        metrics1 = GPUMetrics("GPU1", 1.0, 2.0, 50.0)
        metrics2 = GPUMetrics("GPU1", 1.0, 2.0, 50.0)
        metrics3 = GPUMetrics("GPU2", 1.0, 2.0, 50.0)

        assert metrics1 == metrics2
        assert metrics1 != metrics3
        assert hash(metrics1) == hash(metrics2)  # Immutable objects should hash equally

    def test_gpu_metrics_field_types(self):
        """Test GPUMetrics field type validation."""
        # This tests that the dataclass accepts the expected types
        gpu_metrics = GPUMetrics(
            device_name="RTX 4090",
            memory_allocated_gb=16.5,
            memory_reserved_gb=24.0,
            utilization_percent=68.75,
        )

        assert isinstance(gpu_metrics.device_name, str)
        assert isinstance(gpu_metrics.memory_allocated_gb, float)
        assert isinstance(gpu_metrics.memory_reserved_gb, float)
        assert isinstance(gpu_metrics.utilization_percent, float)


class TestGPUMonitoringIntegration:
    """Integration tests for GPU monitoring (may require actual GPU)."""

    @pytest.mark.integration
    @pytest.mark.requires_gpu
    async def test_real_gpu_monitoring(self):
        """Test monitoring with real GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU available for integration test")

        async with gpu_performance_monitor() as gpu_metrics:
            if gpu_metrics is not None:
                assert isinstance(gpu_metrics.device_name, str)
                assert len(gpu_metrics.device_name) > 0
                assert gpu_metrics.memory_allocated_gb >= 0
                assert gpu_metrics.memory_reserved_gb >= 0
                assert 0 <= gpu_metrics.utilization_percent <= 100
            else:
                pytest.fail("Expected GPU metrics but got None")

    @pytest.mark.integration
    async def test_fallback_behavior_no_gpu(self):
        """Test fallback behavior when no GPU is available."""
        with patch("torch.cuda.is_available", return_value=False):
            async with gpu_performance_monitor() as gpu_metrics:
                assert gpu_metrics is None
