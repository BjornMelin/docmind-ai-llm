"""Enhanced tests for GPU monitor infrastructure - proper mocking and edge cases.

Focuses on areas not fully covered by existing tests:
- Proper torch.cuda mocking without side effects
- Multi-device scenarios
- Device switching and state management
- Memory allocation edge cases
- Performance degradation detection
- Context manager lifecycle
"""

import asyncio
from unittest.mock import Mock, patch

import pytest

from src.core.infrastructure.gpu_monitor import GPUMetrics, gpu_performance_monitor


class TestGPUMonitorMockingIsolation:
    """Test proper mocking isolation for GPU monitor."""

    @pytest.mark.unit
    @patch("torch.cuda.is_available")
    async def test_cuda_unavailable_isolation(self, mock_cuda_available):
        """Test CUDA unavailable scenario with proper isolation."""
        mock_cuda_available.return_value = False

        async with gpu_performance_monitor() as metrics:
            assert metrics is None

        # Verify mock was called
        mock_cuda_available.assert_called_once()

    @pytest.mark.unit
    async def test_cuda_available_with_full_mock_stack(self):
        """Test CUDA available with complete mock stack."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=2684354560),  # 2.5GB
            patch("torch.cuda.memory_reserved", return_value=3221225472),  # 3GB
        ):
            # Mock device properties
            mock_device = Mock()
            mock_device.name = "Test GPU"
            mock_device.total_memory = 10737418240  # 10GB
            mock_props.return_value = mock_device

            async with gpu_performance_monitor() as metrics:
                assert metrics is not None
                assert isinstance(metrics, GPUMetrics)
                assert metrics.device_name == "Test GPU"
                assert abs(metrics.memory_allocated_gb - 2.5) < 0.1
                assert abs(metrics.memory_reserved_gb - 3.0) < 0.1
                assert abs(metrics.utilization_percent - 25.0) < 0.1

    @pytest.mark.unit
    async def test_mock_isolation_between_tests(self):
        """Ensure mocks don't leak between test runs."""
        # First call with CUDA unavailable
        with patch("torch.cuda.is_available", return_value=False):
            async with gpu_performance_monitor() as metrics1:
                result1 = metrics1

        # Second call with CUDA available
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=1073741824),  # 1GB
            patch("torch.cuda.memory_reserved", return_value=2147483648),  # 2GB
        ):
            mock_device = Mock()
            mock_device.name = "Second Test GPU"
            mock_device.total_memory = 4294967296  # 4GB
            mock_props.return_value = mock_device

            async with gpu_performance_monitor() as metrics2:
                result2 = metrics2

        # Results should be independent
        assert result1 is None
        assert result2 is not None
        assert result2.device_name == "Second Test GPU"


class TestGPUMemoryCalculationEdgeCases:
    """Test edge cases in GPU memory calculations."""

    @pytest.mark.unit
    async def test_zero_total_memory_handling(self):
        """Test handling when GPU reports zero total memory."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=0),
            patch("torch.cuda.memory_reserved", return_value=0),
        ):
            mock_device = Mock()
            mock_device.name = "Zero Memory GPU"
            mock_device.total_memory = 0  # Edge case
            mock_props.return_value = mock_device

            # Should handle division by zero gracefully
            with pytest.raises(ZeroDivisionError):
                async with gpu_performance_monitor():
                    pass  # Should raise before yielding

    @pytest.mark.unit
    async def test_memory_exceeds_total_clamping(self):
        """Test memory utilization clamping when allocated > total."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=12884901888),  # 12GB
            patch("torch.cuda.memory_reserved", return_value=12884901888),  # 12GB
        ):
            mock_device = Mock()
            mock_device.name = "Overcommit GPU"
            mock_device.total_memory = 10737418240  # 10GB total
            mock_props.return_value = mock_device

            async with gpu_performance_monitor() as metrics:
                assert metrics is not None
                # Utilization should be clamped at 100%
                assert metrics.utilization_percent == 100.0

    @pytest.mark.unit
    async def test_fractional_memory_precision(self):
        """Test precision with fractional memory values."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=1234567890),  # ~1.15GB
            patch("torch.cuda.memory_reserved", return_value=2345678901),  # ~2.18GB
        ):
            mock_device = Mock()
            mock_device.name = "Precision GPU"
            mock_device.total_memory = 8589934592  # 8GB
            mock_props.return_value = mock_device

            async with gpu_performance_monitor() as metrics:
                assert metrics is not None
                # Check precision (should be ~1.15GB allocated)
                assert 1.14 < metrics.memory_allocated_gb < 1.16
                # Check precision (should be ~2.18GB reserved)
                assert 2.17 < metrics.memory_reserved_gb < 2.19
                # Utilization should be based on allocated vs total
                expected_util = (1234567890 / 8589934592) * 100
                assert abs(metrics.utilization_percent - expected_util) < 0.1


class TestGPUMultiDeviceScenarios:
    """Test multi-device GPU scenarios."""

    @pytest.mark.unit
    async def test_multi_device_current_device_selection(self):
        """Test behavior with multiple GPUs when current device is selected."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=1),  # Second GPU
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated") as mock_allocated,
            patch("torch.cuda.memory_reserved") as mock_reserved,
        ):
            # Mock device properties for device 1
            mock_device = Mock()
            mock_device.name = "GPU Device 1"
            mock_device.total_memory = 16106127360  # 15GB
            mock_props.return_value = mock_device

            # Mock memory calls for device 1
            mock_allocated.return_value = 5368709120  # 5GB
            mock_reserved.return_value = 6442450944  # 6GB

            async with gpu_performance_monitor() as metrics:
                assert metrics is not None
                assert metrics.device_name == "GPU Device 1"

                # Verify calls were made with correct device
                mock_props.assert_called_once_with(1)
                mock_allocated.assert_called_once_with(1)
                mock_reserved.assert_called_once_with(1)

    @pytest.mark.unit
    async def test_device_switch_during_monitoring(self):
        """Test behavior when device changes during monitoring."""
        device_sequence = [0, 1, 0]  # Simulate device switching
        call_count = 0

        def mock_current_device():
            nonlocal call_count
            device = device_sequence[call_count % len(device_sequence)]
            call_count += 1
            return device

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", side_effect=mock_current_device),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=2147483648),  # 2GB
            patch("torch.cuda.memory_reserved", return_value=3221225472),  # 3GB
        ):
            mock_device = Mock()
            mock_device.name = "Switching GPU"
            mock_device.total_memory = 8589934592  # 8GB
            mock_props.return_value = mock_device

            # Each monitoring call should use current device
            async with gpu_performance_monitor() as metrics:
                assert metrics is not None
                assert metrics.device_name == "Switching GPU"


class TestGPUContextManagerLifecycle:
    """Test GPU context manager lifecycle and cleanup."""

    @pytest.mark.unit
    async def test_context_manager_exception_handling(self):
        """Test context manager behavior when exceptions occur."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=1073741824),
            patch("torch.cuda.memory_reserved", return_value=2147483648),
        ):
            mock_device = Mock()
            mock_device.name = "Exception GPU"
            mock_device.total_memory = 4294967296
            mock_props.return_value = mock_device

            # Context manager should handle exceptions properly
            try:
                async with gpu_performance_monitor() as metrics:
                    assert metrics is not None
                    # Simulate exception inside context
                    raise ValueError("Test exception")
            except ValueError:
                pass  # Expected

            # Should be able to use again after exception
            async with gpu_performance_monitor() as metrics:
                assert metrics is not None
                assert metrics.device_name == "Exception GPU"

    @pytest.mark.unit
    async def test_rapid_context_creation_destruction(self):
        """Test rapid creation and destruction of monitoring contexts."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=1073741824),
            patch("torch.cuda.memory_reserved", return_value=2147483648),
        ):
            mock_device = Mock()
            mock_device.name = "Rapid GPU"
            mock_device.total_memory = 4294967296
            mock_props.return_value = mock_device

            # Create and destroy contexts rapidly
            for _i in range(50):
                async with gpu_performance_monitor() as metrics:
                    assert metrics is not None
                    assert metrics.device_name == "Rapid GPU"

    @pytest.mark.unit
    async def test_nested_context_managers(self):
        """Test nested GPU monitoring contexts."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=1073741824),
            patch("torch.cuda.memory_reserved", return_value=2147483648),
        ):
            mock_device = Mock()
            mock_device.name = "Nested GPU"
            mock_device.total_memory = 4294967296
            mock_props.return_value = mock_device

            # Nested context managers should work independently
            async with gpu_performance_monitor() as metrics1:
                assert metrics1 is not None

                async with gpu_performance_monitor() as metrics2:
                    assert metrics2 is not None

                    # Both should have same data (from same mocks)
                    assert metrics1.device_name == metrics2.device_name
                    assert metrics1.memory_allocated_gb == metrics2.memory_allocated_gb


class TestGPUErrorHandlingScenarios:
    """Test various GPU error handling scenarios."""

    @pytest.mark.unit
    async def test_cuda_driver_not_found(self):
        """Test handling when CUDA driver is not found."""
        with patch(
            "torch.cuda.current_device",
            side_effect=RuntimeError("CUDA driver not found"),
        ), patch("torch.cuda.is_available", return_value=True):
            # Should propagate CUDA driver errors
            with pytest.raises(RuntimeError, match="CUDA driver not found"):
                async with gpu_performance_monitor():
                    pass

    @pytest.mark.unit
    async def test_out_of_memory_during_monitoring(self):
        """Test handling when CUDA runs out of memory during monitoring."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch(
                "torch.cuda.memory_allocated",
                side_effect=RuntimeError("CUDA out of memory"),
            ),
            patch("torch.cuda.memory_reserved", return_value=0),
        ):
            mock_device = Mock()
            mock_device.name = "OOM GPU"
            mock_device.total_memory = 4294967296
            mock_props.return_value = mock_device

            # Should propagate CUDA OOM errors
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                async with gpu_performance_monitor():
                    pass

    @pytest.mark.unit
    async def test_device_busy_during_properties_query(self):
        """Test handling when device is busy during properties query."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch(
                "torch.cuda.get_device_properties",
                side_effect=RuntimeError("Device busy"),
            ),
        ):
            # Should propagate device busy errors
            with pytest.raises(RuntimeError, match="Device busy"):
                async with gpu_performance_monitor():
                    pass


class TestGPUPerformanceMonitoring:
    """Test GPU performance monitoring capabilities."""

    @pytest.mark.unit
    async def test_concurrent_monitoring_same_device(self):
        """Test concurrent monitoring of the same device."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=2147483648),
            patch("torch.cuda.memory_reserved", return_value=3221225472),
        ):
            mock_device = Mock()
            mock_device.name = "Concurrent Monitor GPU"
            mock_device.total_memory = 8589934592
            mock_props.return_value = mock_device

            # Run multiple concurrent monitoring operations
            async def monitor():
                async with gpu_performance_monitor() as metrics:
                    assert metrics is not None
                    return metrics

            results = await asyncio.gather(*[monitor() for _ in range(10)])

            # All should succeed and return consistent data
            assert len(results) == 10
            for metrics in results:
                assert metrics.device_name == "Concurrent Monitor GPU"
                assert metrics.memory_allocated_gb == pytest.approx(2.0, rel=1e-1)

    @pytest.mark.unit
    async def test_monitoring_performance_overhead(self):
        """Test monitoring performance overhead."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.memory_allocated", return_value=1073741824),
            patch("torch.cuda.memory_reserved", return_value=2147483648),
        ):
            mock_device = Mock()
            mock_device.name = "Performance GPU"
            mock_device.total_memory = 4294967296
            mock_props.return_value = mock_device

            import time

            # Measure time for 100 monitoring operations
            start_time = time.time()

            for _ in range(100):
                async with gpu_performance_monitor() as metrics:
                    assert metrics is not None

            elapsed = time.time() - start_time

            # Should be fast (under 1 second for 100 operations)
            assert elapsed < 1.0
            # Average per operation should be under 10ms
            assert (elapsed / 100) < 0.01

    @pytest.mark.unit
    async def test_memory_utilization_boundary_conditions(self):
        """Test memory utilization at boundary conditions."""
        test_cases = [
            # (allocated_bytes, total_bytes, expected_util_percent)
            (0, 1073741824, 0.0),  # 0% utilization
            (1073741824, 1073741824, 100.0),  # 100% utilization
            (536870912, 1073741824, 50.0),  # 50% utilization
            (1, 1073741824, 0.0),  # Minimal utilization
            (1073741823, 1073741824, 100.0),  # Nearly full (should round to 100)
        ]

        for allocated, total, expected_util in test_cases:
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.current_device", return_value=0),
                patch("torch.cuda.get_device_properties") as mock_props,
                patch("torch.cuda.memory_allocated", return_value=allocated),
                patch("torch.cuda.memory_reserved", return_value=allocated),
            ):
                mock_device = Mock()
                mock_device.name = "Boundary GPU"
                mock_device.total_memory = total
                mock_props.return_value = mock_device

                async with gpu_performance_monitor() as metrics:
                    assert metrics is not None
                    if expected_util == 0.0:
                        assert (
                            metrics.utilization_percent <= 0.1
                        )  # Allow small rounding
                    elif expected_util == 100.0:
                        assert (
                            metrics.utilization_percent >= 99.9
                        )  # Allow small rounding
                    else:
                        assert abs(metrics.utilization_percent - expected_util) < 1.0
