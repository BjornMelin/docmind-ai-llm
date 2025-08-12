"""Tests for PyTorch native GPU monitoring."""

from unittest.mock import Mock, patch

import pytest

from src.core.infrastructure.gpu_monitor import GPUMetrics, gpu_performance_monitor


class TestGPUMetrics:
    """Test GPUMetrics dataclass."""

    def test_gpu_metrics_creation(self):
        """Test GPUMetrics creation with valid data."""
        metrics = GPUMetrics(
            device_name="NVIDIA GeForce RTX 3080",
            memory_allocated_gb=2.5,
            memory_reserved_gb=3.0,
            utilization_percent=80.0,
        )

        assert metrics.device_name == "NVIDIA GeForce RTX 3080"
        assert metrics.memory_allocated_gb == 2.5
        assert metrics.memory_reserved_gb == 3.0
        assert metrics.utilization_percent == 80.0

    def test_gpu_metrics_immutable(self):
        """Test that GPUMetrics is immutable (frozen=True)."""
        metrics = GPUMetrics("Test", 1.0, 2.0, 50.0)

        with pytest.raises(AttributeError):
            metrics.device_name = "Modified"


@pytest.mark.asyncio
class TestGPUPerformanceMonitor:
    """Test GPU performance monitoring context manager."""

    @patch("torch.cuda.is_available", return_value=False)
    async def test_no_cuda_returns_none(self, mock_cuda_available):
        """Test that context manager returns None when CUDA unavailable."""
        async with gpu_performance_monitor() as metrics:
            assert metrics is None

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

        async with gpu_performance_monitor() as metrics:
            assert metrics is not None
            assert isinstance(metrics, GPUMetrics)
            assert metrics.device_name == "NVIDIA GeForce RTX 3080"
            assert metrics.memory_allocated_gb == pytest.approx(2.5, rel=1e-1)
            assert metrics.memory_reserved_gb == pytest.approx(3.0, rel=1e-1)
            assert metrics.utilization_percent == pytest.approx(25.0, rel=1e-1)

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

        async with gpu_performance_monitor() as metrics:
            assert metrics is not None
            assert metrics.utilization_percent == 100.0
