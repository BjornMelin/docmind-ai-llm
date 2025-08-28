"""Enhanced unit tests for hardware utilities with realistic scenarios.

Focuses on testing actual functionality with minimal mocking, realistic test data,
and comprehensive edge case coverage. Tests behavior rather than mock interactions.
"""

from unittest.mock import patch

import pytest

from src.core.infrastructure.hardware_utils import (
    BYTES_TO_GB_FACTOR,
    CPU_BATCH_SIZES,
    HIGH_END_BATCH_SIZES,
    detect_hardware,
    get_optimal_providers,
    get_recommended_batch_size,
)

# Check for required dependencies
try:
    import torch
except ImportError:
    pytest.skip("torch not available", allow_module_level=True)


class TestDetectHardware:
    """Test hardware detection functionality with realistic scenarios."""

    def test_detect_hardware_basic_structure(self):
        """Test that detect_hardware returns expected structure regardless of hardware.

        Ensures basic structure is returned regardless of underlying hardware.
        """
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
        assert result["gpu_device_count"] >= 0
        assert result["cpu_cores"] >= 1
        assert len(result["fastembed_providers"]) >= 1

    @pytest.mark.integration
    def test_detect_hardware_real_system(self):
        """Test hardware detection with real system hardware."""
        result = detect_hardware()

        # Test realistic behavior
        if result["cuda_available"]:
            assert result["gpu_device_count"] > 0
            assert result["gpu_name"] != "Unknown"
            assert "CUDAExecutionProvider" in result["fastembed_providers"]
            if result["vram_total_gb"]:
                assert result["vram_total_gb"] > 0
        else:
            assert result["gpu_device_count"] == 0
            assert result["gpu_name"] == "Unknown"
            assert result["fastembed_providers"] == ["CPUExecutionProvider"]

        # CPU should always be detected
        assert result["cpu_cores"] >= 1
        assert "CPUExecutionProvider" in result["fastembed_providers"]

    def test_cuda_error_handling(self):
        """Test error handling when CUDA operations fail."""
        with patch("torch.cuda.is_available", side_effect=RuntimeError("CUDA error")):
            result = detect_hardware()

            # Should fall back gracefully
            assert result["cuda_available"] is False
            assert result["gpu_name"] == "Unknown"
            assert result["fastembed_providers"] == ["CPUExecutionProvider"]
            assert result["cpu_cores"] >= 1

    def test_memory_calculation_accuracy(self):
        """Test that memory calculations are accurate."""
        # Test the constant is correct
        assert BYTES_TO_GB_FACTOR == 1024**3

        # Test with known values
        test_bytes = 10737418240  # 10 GB
        expected_gb = test_bytes / BYTES_TO_GB_FACTOR
        assert expected_gb == 10.0

    def test_vram_calculation_edge_cases(self):
        """Test VRAM calculation with various edge cases."""
        # Test various memory sizes that should convert correctly
        test_cases = [
            (1073741824, 1.0),  # 1 GB
            (2147483648, 2.0),  # 2 GB
            (10737418240, 10.0),  # 10 GB
            (17179869184, 16.0),  # 16 GB
        ]

        for bytes_val, expected_gb in test_cases:
            calculated_gb = bytes_val / BYTES_TO_GB_FACTOR
            assert abs(calculated_gb - expected_gb) < 0.1

    def test_detect_hardware_consistency(self):
        """Test that detect_hardware returns consistent results across calls."""
        result1 = detect_hardware()
        result2 = detect_hardware()

        # Results should be identical for static hardware
        # (Note: Available VRAM might vary slightly in real systems)
        assert result1["cuda_available"] == result2["cuda_available"]
        assert result1["gpu_name"] == result2["gpu_name"]
        assert result1["gpu_device_count"] == result2["gpu_device_count"]
        assert result1["cpu_cores"] == result2["cpu_cores"]
        assert result1["fastembed_providers"] == result2["fastembed_providers"]

    def test_error_handling_robustness(self):
        """Test robustness when various errors occur."""
        # Test when os.cpu_count returns None
        with patch("os.cpu_count", return_value=None):
            result = detect_hardware()
            assert result["cpu_cores"] >= 1  # Should default to 1

        # Test when CUDA check raises unexpected error
        with patch("torch.cuda.is_available", side_effect=ImportError("No CUDA")):
            result = detect_hardware()
            assert result["cuda_available"] is False
            assert result["cpu_cores"] >= 1  # Should still work


class TestGetOptimalProviders:
    """Test optimal execution provider selection with realistic scenarios."""

    def test_get_optimal_providers_force_cpu_always_works(self):
        """Test that force_cpu=True always returns CPU provider."""
        result = get_optimal_providers(force_cpu=True)
        assert result == ["CPUExecutionProvider"]

    @pytest.mark.integration
    def test_get_optimal_providers_real_system(self):
        """Test provider selection with real system."""
        # Test default behavior
        providers = get_optimal_providers()
        assert isinstance(providers, list)
        assert len(providers) >= 1
        assert "CPUExecutionProvider" in providers

        # CUDA provider should be first if available
        if torch.cuda.is_available():
            assert providers[0] == "CUDAExecutionProvider"
            assert len(providers) == 2
        else:
            assert providers == ["CPUExecutionProvider"]

        # Force CPU should work regardless of hardware
        cpu_only = get_optimal_providers(force_cpu=True)
        assert cpu_only == ["CPUExecutionProvider"]

    def test_provider_selection_logic(self):
        """Test provider selection logic with various scenarios."""
        # Mock scenarios
        with patch("torch.cuda.is_available", return_value=True):
            result = get_optimal_providers(force_cpu=False)
            assert "CUDAExecutionProvider" in result
            assert "CPUExecutionProvider" in result

        with patch("torch.cuda.is_available", return_value=False):
            result = get_optimal_providers(force_cpu=False)
            assert result == ["CPUExecutionProvider"]


class TestGetRecommendedBatchSize:
    """Test batch size recommendation with realistic data and configurations."""

    def test_batch_size_constants_are_valid(self):
        """Test that batch size constants are reasonable."""
        # Verify CPU batch sizes are reasonable
        assert all(isinstance(v, int) and v > 0 for v in CPU_BATCH_SIZES.values())
        assert all(isinstance(v, int) and v > 0 for v in HIGH_END_BATCH_SIZES.values())

        # Test model type coverage
        expected_types = {"embedding", "llm", "vision"}
        assert set(CPU_BATCH_SIZES.keys()) == expected_types
        assert set(HIGH_END_BATCH_SIZES.keys()) == expected_types

    def test_cpu_batch_size_selection(self):
        """Test CPU batch size selection with realistic scenarios."""
        with patch("torch.cuda.is_available", return_value=False):
            # Test known model types
            for model_type, expected in CPU_BATCH_SIZES.items():
                result = get_recommended_batch_size(model_type)
                assert result == expected

            # Test fallback for unknown types
            result = get_recommended_batch_size("unknown_type")
            assert result == 8  # DEFAULT_BATCH_SIZE_FALLBACK

            # Test edge cases
            assert get_recommended_batch_size("") == 8
            assert get_recommended_batch_size(None) == 8

    def test_batch_size_gpu_tiers_logic(self):
        """Test batch size logic for different GPU memory tiers."""
        # Test thresholds are reasonable
        from src.core.infrastructure.hardware_utils import (
            ENTRY_LEVEL_VRAM_THRESHOLD,
            HIGH_END_VRAM_THRESHOLD,
            MID_RANGE_VRAM_THRESHOLD,
        )

        assert HIGH_END_VRAM_THRESHOLD >= MID_RANGE_VRAM_THRESHOLD
        assert MID_RANGE_VRAM_THRESHOLD >= ENTRY_LEVEL_VRAM_THRESHOLD
        assert ENTRY_LEVEL_VRAM_THRESHOLD > 0

        # Test that batch sizes increase with GPU tier
        for model_type in ["embedding", "llm", "vision"]:
            cpu_size = CPU_BATCH_SIZES[model_type]
            high_end_size = HIGH_END_BATCH_SIZES[model_type]
            assert high_end_size >= cpu_size  # GPU should be >= CPU

    def test_gpu_error_fallback_behavior(self):
        """Test that GPU errors fall back to CPU batch sizes."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "torch.cuda.get_device_properties",
                side_effect=RuntimeError("GPU error"),
            ),
        ):
            # Should fall back to CPU batch sizes when GPU properties fail
            for model_type, expected_cpu_size in CPU_BATCH_SIZES.items():
                result = get_recommended_batch_size(model_type)
                assert result == expected_cpu_size

    @pytest.mark.integration
    def test_batch_size_with_real_hardware(self):
        """Test batch size recommendations with real hardware."""
        for model_type in ["embedding", "llm", "vision", "unknown"]:
            batch_size = get_recommended_batch_size(model_type)
            assert isinstance(batch_size, int)
            assert 1 <= batch_size <= 128  # Reasonable range

        # Default parameter test
        default_result = get_recommended_batch_size()
        embedding_result = get_recommended_batch_size("embedding")
        assert default_result == embedding_result


class TestHardwareUtilsEdgeCases:
    """Test edge cases and realistic error scenarios."""

    def test_batch_size_model_type_variations(self):
        """Test batch size with various model type inputs."""
        with patch("torch.cuda.is_available", return_value=False):
            # Test case insensitivity isn't implemented (should return default)
            result = get_recommended_batch_size("EMBEDDING")
            assert result == 8  # Should use default since it's not "embedding"

            # Test partial matches don't work
            result = get_recommended_batch_size("embed")
            assert result == 8  # Should use default

            # Test whitespace doesn't work
            result = get_recommended_batch_size(" embedding ")
            assert result == 8  # Should use default

    def test_function_parameter_defaults(self):
        """Test that function parameters have sensible defaults."""
        # Test default model type
        default_result = get_recommended_batch_size()
        embedding_result = get_recommended_batch_size("embedding")
        assert default_result == embedding_result

        # Test default force_cpu parameter
        providers1 = get_optimal_providers()
        providers2 = get_optimal_providers(force_cpu=False)
        assert providers1 == providers2


class TestHardwareUtilsPerformance:
    """Test performance characteristics of hardware utilities."""

    @pytest.mark.performance
    def test_hardware_detection_performance(self):
        """Test that hardware detection completes quickly."""
        import time

        start_time = time.time()
        for _ in range(10):
            result = detect_hardware()
            assert isinstance(result, dict)
        elapsed_time = time.time() - start_time

        # Should complete 10 detections in under 500ms
        assert elapsed_time < 0.5

    @pytest.mark.performance
    def test_provider_selection_performance(self):
        """Test that provider selection is very fast."""
        import time

        start_time = time.time()
        for _ in range(1000):
            providers = get_optimal_providers()
            assert isinstance(providers, list)
        elapsed_time = time.time() - start_time

        # Should complete 1000 selections in under 100ms
        assert elapsed_time < 0.1

    @pytest.mark.performance
    def test_batch_size_calculation_performance(self):
        """Test that batch size calculation is fast."""
        import time

        model_types = ["embedding", "llm", "vision", "unknown"]
        start_time = time.time()
        for _ in range(100):
            for model_type in model_types:
                batch_size = get_recommended_batch_size(model_type)
                assert isinstance(batch_size, int)
        elapsed_time = time.time() - start_time

        # Should complete 400 calculations in under 200ms
        assert elapsed_time < 0.2

    def test_all_functions_handle_none_gracefully(self):
        """Test that all functions handle None/invalid inputs gracefully."""
        # These should not crash
        result = get_recommended_batch_size(None)
        assert isinstance(result, int)

        result = get_recommended_batch_size("")
        assert isinstance(result, int)

        providers = get_optimal_providers(force_cpu=None)
        assert isinstance(providers, list)

        # Hardware detection should always work
        hardware = detect_hardware()
        assert isinstance(hardware, dict)
