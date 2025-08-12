"""Comprehensive tests for hardware utilities.

This module provides comprehensive test coverage for hardware detection functionality,
including GPU/CPU detection, batch size calculation, provider selection, and edge cases.
"""

from unittest.mock import Mock, patch

import pytest

from src.core.infrastructure.hardware_utils import (
    detect_hardware,
    get_optimal_providers,
    get_recommended_batch_size,
)


class TestDetectHardware:
    """Test hardware detection functionality."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", return_value=0)
    @patch("os.cpu_count", return_value=8)
    def test_detect_hardware_with_cuda(
        self,
        mock_cpu_count,
        mock_memory_allocated,
        mock_props,
        mock_device_count,
        mock_cuda_available,
    ):
        """Test hardware detection with CUDA available."""
        # Mock GPU device properties
        mock_device = Mock()
        mock_device.name = "NVIDIA GeForce RTX 3080"
        mock_device.total_memory = 10737418240  # 10 GB
        mock_device.major = 8
        mock_device.minor = 6
        mock_props.return_value = mock_device

        result = detect_hardware()

        assert result["cuda_available"] is True
        assert result["gpu_name"] == "NVIDIA GeForce RTX 3080"
        assert result["vram_total_gb"] == 10.0
        assert result["vram_available_gb"] == 10.0  # No memory allocated
        assert result["gpu_compute_capability"] == (8, 6)
        assert result["gpu_device_count"] == 2
        assert result["fastembed_providers"] == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert result["cpu_cores"] == 8
        assert result["cpu_threads"] == 8

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.cuda.device_count", return_value=0)
    @patch("os.cpu_count", return_value=4)
    def test_detect_hardware_without_cuda(
        self, mock_cpu_count, mock_device_count, mock_cuda_available
    ):
        """Test hardware detection without CUDA."""
        result = detect_hardware()

        assert result["cuda_available"] is False
        assert result["gpu_name"] == "Unknown"
        assert result["vram_total_gb"] is None
        assert result["vram_available_gb"] is None
        assert result["gpu_compute_capability"] is None
        assert result["gpu_device_count"] == 0
        assert result["fastembed_providers"] == ["CPUExecutionProvider"]
        assert result["cpu_cores"] == 4
        assert result["cpu_threads"] == 4

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", return_value=2147483648)  # 2 GB allocated
    @patch("os.cpu_count", return_value=16)
    def test_detect_hardware_with_allocated_memory(
        self,
        mock_cpu_count,
        mock_memory_allocated,
        mock_props,
        mock_device_count,
        mock_cuda_available,
    ):
        """Test hardware detection with some GPU memory already allocated."""
        mock_device = Mock()
        mock_device.name = "NVIDIA RTX A4000"
        mock_device.total_memory = 17179869184  # 16 GB
        mock_device.major = 8
        mock_device.minor = 6
        mock_props.return_value = mock_device

        result = detect_hardware()

        assert result["cuda_available"] is True
        assert result["gpu_name"] == "NVIDIA RTX A4000"
        assert result["vram_total_gb"] == 16.0
        assert result["vram_available_gb"] == 14.0  # 16 - 2 = 14 GB available
        assert result["gpu_compute_capability"] == (8, 6)
        assert result["gpu_device_count"] == 1

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", side_effect=RuntimeError("CUDA error"))
    @patch("os.cpu_count", return_value=12)
    def test_detect_hardware_memory_allocation_error(
        self,
        mock_cpu_count,
        mock_memory_allocated,
        mock_props,
        mock_device_count,
        mock_cuda_available,
    ):
        """Test hardware detection when memory allocation query fails."""
        mock_device = Mock()
        mock_device.name = "Test GPU"
        mock_device.total_memory = 8589934592  # 8 GB
        mock_device.major = 7
        mock_device.minor = 5
        mock_props.return_value = mock_device

        result = detect_hardware()

        # Should fall back to total memory when allocation query fails
        assert result["cuda_available"] is True
        assert result["gpu_name"] == "Test GPU"
        assert result["vram_total_gb"] == 8.0
        assert result["vram_available_gb"] == 8.0  # Falls back to total
        assert result["gpu_compute_capability"] == (7, 5)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=4)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", return_value=0)
    @patch("os.cpu_count", return_value=32)
    def test_detect_hardware_multiple_gpus(
        self,
        mock_cpu_count,
        mock_memory_allocated,
        mock_props,
        mock_device_count,
        mock_cuda_available,
    ):
        """Test hardware detection with multiple GPUs."""
        mock_device = Mock()
        mock_device.name = "NVIDIA A100-SXM4-80GB"
        mock_device.total_memory = 85899345920  # 80 GB
        mock_device.major = 8
        mock_device.minor = 0
        mock_props.return_value = mock_device

        result = detect_hardware()

        assert result["cuda_available"] is True
        assert result["gpu_device_count"] == 4  # Multiple GPUs detected
        assert result["gpu_name"] == "NVIDIA A100-SXM4-80GB"
        assert result["vram_total_gb"] == 80.0
        assert result["cpu_cores"] == 32

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", return_value=0)
    @patch("os.cpu_count", return_value=None)
    def test_detect_hardware_cpu_count_none(
        self,
        mock_cpu_count,
        mock_memory_allocated,
        mock_props,
        mock_device_count,
        mock_cuda_available,
    ):
        """Test hardware detection when CPU count is None."""
        mock_device = Mock()
        mock_device.name = "Test GPU"
        mock_device.total_memory = 4294967296  # 4 GB
        mock_device.major = 6
        mock_device.minor = 1
        mock_props.return_value = mock_device

        result = detect_hardware()

        # Should default to 1 when os.cpu_count() returns None
        assert result["cpu_cores"] == 1
        assert result["cpu_threads"] == 1

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", return_value=0)
    @patch("os.cpu_count", return_value=64)
    def test_detect_hardware_edge_case_values(
        self,
        mock_cpu_count,
        mock_memory_allocated,
        mock_props,
        mock_device_count,
        mock_cuda_available,
    ):
        """Test hardware detection with edge case values."""
        # Test with very small GPU memory
        mock_device = Mock()
        mock_device.name = "Low Memory GPU"
        mock_device.total_memory = 1073741824  # 1 GB
        mock_device.major = 5
        mock_device.minor = 2
        mock_props.return_value = mock_device

        result = detect_hardware()

        assert result["vram_total_gb"] == 1.0
        assert result["gpu_compute_capability"] == (5, 2)
        assert result["cpu_cores"] == 64

    def test_detect_hardware_return_structure(self):
        """Test that detect_hardware returns expected structure."""
        with patch("torch.cuda.is_available", return_value=False):
            result = detect_hardware()

            # Verify all expected keys are present
            expected_keys = {
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
            assert set(result.keys()) == expected_keys

            # Verify types
            assert isinstance(result["cuda_available"], bool)
            assert isinstance(result["gpu_name"], str)
            assert isinstance(result["gpu_device_count"], int)
            assert isinstance(result["fastembed_providers"], list)
            assert isinstance(result["cpu_cores"], int)
            assert isinstance(result["cpu_threads"], int)


class TestGetOptimalProviders:
    """Test optimal execution provider selection."""

    @patch("torch.cuda.is_available", return_value=True)
    def test_get_optimal_providers_cuda_available(self, mock_cuda_available):
        """Test provider selection when CUDA is available."""
        result = get_optimal_providers()
        assert result == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_optimal_providers_no_cuda(self, mock_cuda_available):
        """Test provider selection when CUDA is not available."""
        result = get_optimal_providers()
        assert result == ["CPUExecutionProvider"]

    @patch("torch.cuda.is_available", return_value=True)
    def test_get_optimal_providers_force_cpu(self, mock_cuda_available):
        """Test provider selection when forcing CPU-only execution."""
        result = get_optimal_providers(force_cpu=True)
        assert result == ["CPUExecutionProvider"]

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_optimal_providers_force_cpu_no_cuda(self, mock_cuda_available):
        """Test provider selection when forcing CPU and no CUDA available."""
        result = get_optimal_providers(force_cpu=True)
        assert result == ["CPUExecutionProvider"]

    def test_get_optimal_providers_parameters(self):
        """Test that get_optimal_providers accepts expected parameters."""
        # Test with default parameters
        with patch("torch.cuda.is_available", return_value=True):
            result1 = get_optimal_providers()
            assert isinstance(result1, list)

        # Test with force_cpu=False
        with patch("torch.cuda.is_available", return_value=True):
            result2 = get_optimal_providers(force_cpu=False)
            assert isinstance(result2, list)

        # Test with force_cpu=True
        with patch("torch.cuda.is_available", return_value=True):
            result3 = get_optimal_providers(force_cpu=True)
            assert result3 == ["CPUExecutionProvider"]


class TestGetRecommendedBatchSize:
    """Test batch size recommendation logic."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_recommended_batch_size_cpu_only(self, mock_cuda_available):
        """Test batch size recommendations for CPU-only execution."""
        # Test embedding model
        result = get_recommended_batch_size("embedding")
        assert result == 16

        # Test LLM model
        result = get_recommended_batch_size("llm")
        assert result == 1

        # Test vision model
        result = get_recommended_batch_size("vision")
        assert result == 4

        # Test unknown model type (should return default)
        result = get_recommended_batch_size("unknown")
        assert result == 8

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_get_recommended_batch_size_high_end_gpu(
        self, mock_props, mock_cuda_available
    ):
        """Test batch size recommendations for high-end GPU (16+ GB)."""
        mock_device = Mock()
        mock_device.total_memory = 21474836480  # 20 GB
        mock_props.return_value = mock_device

        # Test embedding model
        result = get_recommended_batch_size("embedding")
        assert result == 128

        # Test LLM model
        result = get_recommended_batch_size("llm")
        assert result == 8

        # Test vision model
        result = get_recommended_batch_size("vision")
        assert result == 32

        # Test unknown model type
        result = get_recommended_batch_size("unknown")
        assert result == 64

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_get_recommended_batch_size_mid_range_gpu(
        self, mock_props, mock_cuda_available
    ):
        """Test batch size recommendations for mid-range GPU (8-16 GB)."""
        mock_device = Mock()
        mock_device.total_memory = 10737418240  # 10 GB
        mock_props.return_value = mock_device

        result = get_recommended_batch_size("embedding")
        assert result == 64

        result = get_recommended_batch_size("llm")
        assert result == 4

        result = get_recommended_batch_size("vision")
        assert result == 16

        result = get_recommended_batch_size("unknown")
        assert result == 32

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_get_recommended_batch_size_entry_level_gpu(
        self, mock_props, mock_cuda_available
    ):
        """Test batch size recommendations for entry-level GPU (4-8 GB)."""
        mock_device = Mock()
        mock_device.total_memory = 6442450944  # 6 GB
        mock_props.return_value = mock_device

        result = get_recommended_batch_size("embedding")
        assert result == 32

        result = get_recommended_batch_size("llm")
        assert result == 2

        result = get_recommended_batch_size("vision")
        assert result == 8

        result = get_recommended_batch_size("unknown")
        assert result == 16

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_get_recommended_batch_size_low_memory_gpu(
        self, mock_props, mock_cuda_available
    ):
        """Test batch size recommendations for low memory GPU (<4 GB)."""
        mock_device = Mock()
        mock_device.total_memory = 2147483648  # 2 GB
        mock_props.return_value = mock_device

        result = get_recommended_batch_size("embedding")
        assert result == 16

        result = get_recommended_batch_size("llm")
        assert result == 1

        result = get_recommended_batch_size("vision")
        assert result == 4

        result = get_recommended_batch_size("unknown")
        assert result == 8

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties", side_effect=RuntimeError("GPU error"))
    def test_get_recommended_batch_size_gpu_error(
        self, mock_props, mock_cuda_available
    ):
        """Test batch size recommendations when GPU properties query fails."""
        # Should fall back to 4GB assumption
        result = get_recommended_batch_size("embedding")
        assert result == 32  # Same as 4GB GPU

        result = get_recommended_batch_size("llm")
        assert result == 2

        result = get_recommended_batch_size("vision")
        assert result == 8

    def test_get_recommended_batch_size_default_parameter(self):
        """Test default parameter for model type."""
        with patch("torch.cuda.is_available", return_value=False):
            # Should default to "embedding"
            result = get_recommended_batch_size()
            assert result == 16  # CPU embedding default

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_get_recommended_batch_size_boundary_conditions(
        self, mock_props, mock_cuda_available
    ):
        """Test batch size recommendations at memory boundaries."""
        # Test exactly 16 GB (high-end threshold)
        mock_device = Mock()
        mock_device.total_memory = 17179869184  # Exactly 16 GB
        mock_props.return_value = mock_device

        result = get_recommended_batch_size("embedding")
        assert result == 128  # Should use high-end settings

        # Test exactly 8 GB (mid-range threshold)
        mock_device.total_memory = 8589934592  # Exactly 8 GB
        mock_props.return_value = mock_device

        result = get_recommended_batch_size("embedding")
        assert result == 64  # Should use mid-range settings

        # Test exactly 4 GB (entry-level threshold)
        mock_device.total_memory = 4294967296  # Exactly 4 GB
        mock_props.return_value = mock_device

        result = get_recommended_batch_size("embedding")
        assert result == 32  # Should use entry-level settings


class TestHardwareUtilsEdgeCases:
    """Test edge cases and error conditions."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=0)
    @patch(
        "torch.cuda.get_device_properties",
        side_effect=AssertionError("Invalid device id"),
    )
    def test_inconsistent_cuda_state(
        self, mock_props, mock_device_count, mock_cuda_available
    ):
        """Test handling of inconsistent CUDA state."""
        # CUDA available but device count is 0 (shouldn't happen but test robustness)
        # This should raise an exception when trying to get device properties
        with pytest.raises(AssertionError, match="Invalid device id"):
            detect_hardware()

    def test_extreme_batch_sizes(self):
        """Test batch size calculations with extreme GPU memory values."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties") as mock_props,
        ):
            # Test with extremely large GPU memory (1TB - hypothetical future GPU)
            mock_device = Mock()
            mock_device.total_memory = 1099511627776  # 1TB
            mock_props.return_value = mock_device

            result = get_recommended_batch_size("embedding")
            assert result == 128  # Should cap at high-end recommendation

            # Test with extremely small GPU memory (512MB - old GPU)
            mock_device.total_memory = 536870912  # 512MB
            mock_props.return_value = mock_device

            result = get_recommended_batch_size("embedding")
            assert result == 16  # Should use low-memory fallback

    def test_empty_string_model_type(self):
        """Test batch size recommendation with empty string model type."""
        with patch("torch.cuda.is_available", return_value=False):
            result = get_recommended_batch_size("")
            assert result == 8  # Should return default value

    def test_none_model_type(self):
        """Test batch size recommendation with None model type."""
        with patch("torch.cuda.is_available", return_value=False):
            result = get_recommended_batch_size(None)
            assert result == 8  # Should return default value

    @patch(
        "torch.cuda.is_available",
        side_effect=RuntimeError("CUDA initialization failed"),
    )
    @patch("os.cpu_count", return_value=4)
    def test_cuda_initialization_error(self, mock_cpu_count, mock_cuda_available):
        """Test handling of CUDA initialization errors."""
        with pytest.raises(RuntimeError, match="CUDA initialization failed"):
            detect_hardware()


class TestHardwareUtilsIntegration:
    """Integration tests for hardware utilities."""

    @pytest.mark.integration
    def test_real_hardware_detection(self):
        """Test hardware detection with real system (integration test)."""
        result = detect_hardware()

        # Basic structure validation
        assert isinstance(result, dict)
        assert "cuda_available" in result
        assert "cpu_cores" in result
        assert "cpu_threads" in result

        # Validate realistic values
        assert result["cpu_cores"] >= 1
        assert result["cpu_threads"] >= 1
        assert isinstance(result["fastembed_providers"], list)
        assert len(result["fastembed_providers"]) >= 1

        if result["cuda_available"]:
            assert result["gpu_device_count"] >= 1
            assert isinstance(result["gpu_name"], str)
            assert len(result["gpu_name"]) > 0
            assert "CUDAExecutionProvider" in result["fastembed_providers"]
        else:
            assert result["gpu_device_count"] == 0
            assert result["gpu_name"] == "Unknown"
            assert result["fastembed_providers"] == ["CPUExecutionProvider"]

    @pytest.mark.integration
    def test_real_provider_selection(self):
        """Test provider selection with real system."""
        providers = get_optimal_providers()
        assert isinstance(providers, list)
        assert len(providers) >= 1
        assert "CPUExecutionProvider" in providers

        # Test force CPU
        cpu_providers = get_optimal_providers(force_cpu=True)
        assert cpu_providers == ["CPUExecutionProvider"]

    @pytest.mark.integration
    def test_real_batch_size_recommendations(self):
        """Test batch size recommendations with real system."""
        for model_type in ["embedding", "llm", "vision", "unknown"]:
            batch_size = get_recommended_batch_size(model_type)
            assert isinstance(batch_size, int)
            assert batch_size >= 1
            assert batch_size <= 128  # Reasonable upper bound

    @pytest.mark.integration
    def test_hardware_consistency(self):
        """Test that hardware detection results are consistent across calls."""
        result1 = detect_hardware()
        result2 = detect_hardware()

        # Results should be identical for static hardware
        assert result1 == result2

    @pytest.mark.performance
    def test_hardware_detection_performance(self):
        """Test that hardware detection completes quickly."""
        import time

        start_time = time.time()
        for _ in range(10):
            detect_hardware()
        elapsed_time = time.time() - start_time

        # Should complete 10 detections in under 100ms
        assert elapsed_time < 0.1

    @pytest.mark.performance
    def test_provider_selection_performance(self):
        """Test that provider selection is fast."""
        import time

        start_time = time.time()
        for _ in range(100):
            get_optimal_providers()
        elapsed_time = time.time() - start_time

        # Should complete 100 selections in under 50ms
        assert elapsed_time < 0.05

    @pytest.mark.performance
    def test_batch_size_calculation_performance(self):
        """Test that batch size calculation is fast."""
        import time

        model_types = ["embedding", "llm", "vision", "unknown"]
        start_time = time.time()
        for _ in range(50):
            for model_type in model_types:
                get_recommended_batch_size(model_type)
        elapsed_time = time.time() - start_time

        # Should complete 200 calculations in under 100ms
        assert elapsed_time < 0.1
