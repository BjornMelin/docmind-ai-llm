"""Unit tests for core infrastructure modules.

This test suite provides extensive coverage for the core infrastructure components:
- gpu_monitor.py: GPU performance monitoring
- hardware_utils.py: Hardware detection and optimization
- spacy_manager.py: SpaCy model management

Test Strategy:
- Mock hardware dependencies (torch.cuda, os)
- Test both available and unavailable hardware scenarios
- Cover error handling and edge cases
- Test thread safety and concurrent access
- Focus on business logic rather than hardware specifics

Markers:
- @pytest.mark.unit: Fast synchronous logic tests
- @pytest.mark.asyncio: Async GPU monitoring tests
"""

import threading
import time
from unittest.mock import Mock, patch

import pytest

# GPU Monitor imports and tests
from src.core.infrastructure.gpu_monitor import (
    BYTES_TO_GB_FACTOR,
    MAX_UTILIZATION_PERCENT,
    PERCENT_CONVERSION_FACTOR,
    GPUMetrics,
    gpu_performance_monitor,
)
from src.core.infrastructure.hardware_utils import (
    BYTES_TO_GB_FACTOR as HW_BYTES_TO_GB_FACTOR,
)

# Hardware Utils imports and tests
from src.core.infrastructure.hardware_utils import (
    DEFAULT_BATCH_SIZE_FALLBACK,
    DEFAULT_CPU_CORES,
    ENTRY_LEVEL_VRAM_THRESHOLD,
    HIGH_END_VRAM_THRESHOLD,
    MID_RANGE_VRAM_THRESHOLD,
    detect_hardware,
    get_optimal_providers,
    get_recommended_batch_size,
)

# SpaCy Manager imports and tests
from src.core.infrastructure.spacy_manager import (
    SpacyManager,
    get_spacy_manager,
)


@pytest.mark.unit
class TestGPUMetrics:
    """Test GPUMetrics dataclass."""

    def test_gpu_metrics_creation(self):
        """Test GPUMetrics dataclass creation."""
        metrics = GPUMetrics(
            device_name="NVIDIA RTX 4090",
            memory_allocated_gb=8.5,
            memory_reserved_gb=12.0,
            utilization_percent=70.8,
        )

        assert metrics.device_name == "NVIDIA RTX 4090"
        assert metrics.memory_allocated_gb == 8.5
        assert metrics.memory_reserved_gb == 12.0
        assert metrics.utilization_percent == 70.8

    def test_gpu_metrics_immutable(self):
        """Test that GPUMetrics is immutable (frozen)."""
        metrics = GPUMetrics("GPU", 1.0, 2.0, 50.0)

        with pytest.raises(AttributeError):
            metrics.device_name = "Modified GPU"

    def test_gpu_metrics_equality(self):
        """Test GPUMetrics equality comparison."""
        metrics1 = GPUMetrics("GPU", 1.0, 2.0, 50.0)
        metrics2 = GPUMetrics("GPU", 1.0, 2.0, 50.0)
        metrics3 = GPUMetrics("Different GPU", 1.0, 2.0, 50.0)

        assert metrics1 == metrics2
        assert metrics1 != metrics3

    def test_gpu_metrics_constants(self):
        """Test GPU monitoring constants."""
        assert BYTES_TO_GB_FACTOR == 1024**3
        assert PERCENT_CONVERSION_FACTOR == 100
        assert MAX_UTILIZATION_PERCENT == 100.0


@pytest.mark.asyncio
class TestGPUPerformanceMonitor:
    """Test gpu_performance_monitor async context manager."""

    async def test_gpu_monitor_no_cuda(self):
        """Test GPU monitor when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            async with gpu_performance_monitor() as metrics:
                assert metrics is None

    async def test_gpu_monitor_with_cuda(self):
        """Test GPU monitor when CUDA is available."""
        # Mock torch.cuda methods
        # Mock device properties
        mock_props = Mock()
        mock_props.name = "NVIDIA RTX 4090"
        mock_props.total_memory = 24 * BYTES_TO_GB_FACTOR

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.memory_allocated", return_value=8 * BYTES_TO_GB_FACTOR),
            patch("torch.cuda.memory_reserved", return_value=12 * BYTES_TO_GB_FACTOR),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            async with gpu_performance_monitor() as metrics:
                assert metrics is not None
                assert isinstance(metrics, GPUMetrics)
                assert metrics.device_name == "NVIDIA RTX 4090"
                assert metrics.memory_allocated_gb == 8.0
                assert metrics.memory_reserved_gb == 12.0
                assert 0 <= metrics.utilization_percent <= 100.0

    async def test_gpu_monitor_utilization_calculation(self):
        """Test GPU utilization percentage calculation."""
        allocated_gb = 6
        total_gb = 24
        expected_utilization = (allocated_gb / total_gb) * 100

        mock_props = Mock()
        mock_props.name = "Test GPU"
        mock_props.total_memory = total_gb * BYTES_TO_GB_FACTOR

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch(
                "torch.cuda.memory_allocated",
                return_value=allocated_gb * BYTES_TO_GB_FACTOR,
            ),
            patch("torch.cuda.memory_reserved", return_value=8 * BYTES_TO_GB_FACTOR),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            async with gpu_performance_monitor() as metrics:
                assert abs(metrics.utilization_percent - expected_utilization) < 0.1

    async def test_gpu_monitor_max_utilization_cap(self):
        """Test that utilization is capped at 100%."""
        # Create scenario where calculated utilization would exceed 100%
        allocated_gb = 30  # More than total
        total_gb = 24

        mock_props = Mock()
        mock_props.name = "Test GPU"
        mock_props.total_memory = total_gb * BYTES_TO_GB_FACTOR

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch(
                "torch.cuda.memory_allocated",
                return_value=allocated_gb * BYTES_TO_GB_FACTOR,
            ),
            patch("torch.cuda.memory_reserved", return_value=32 * BYTES_TO_GB_FACTOR),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            async with gpu_performance_monitor() as metrics:
                assert metrics.utilization_percent == MAX_UTILIZATION_PERCENT


@pytest.mark.unit
class TestHardwareDetection:
    """Test hardware detection utilities."""

    def test_detect_hardware_no_cuda(self):
        """Test hardware detection when CUDA is not available."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("os.cpu_count", return_value=8),
        ):
            hardware = detect_hardware()

            assert hardware["cuda_available"] is False
            assert hardware["gpu_name"] == "Unknown"
            assert hardware["vram_total_gb"] is None
            assert hardware["vram_available_gb"] is None
            assert hardware["gpu_compute_capability"] is None
            assert hardware["gpu_device_count"] == 0
            assert hardware["fastembed_providers"] == ["CPUExecutionProvider"]
            assert hardware["cpu_cores"] == 8
            assert hardware["cpu_threads"] == 8

    def test_detect_hardware_with_cuda(self):
        """Test hardware detection when CUDA is available."""
        mock_props = Mock()
        mock_props.name = "NVIDIA RTX 4090"
        mock_props.total_memory = 24 * HW_BYTES_TO_GB_FACTOR
        mock_props.major = 8
        mock_props.minor = 9

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
            patch(
                "torch.cuda.memory_allocated", return_value=6 * HW_BYTES_TO_GB_FACTOR
            ),
            patch("os.cpu_count", return_value=16),
        ):
            hardware = detect_hardware()

            assert hardware["cuda_available"] is True
            assert hardware["gpu_name"] == "NVIDIA RTX 4090"
            assert hardware["vram_total_gb"] == 24.0
            assert hardware["vram_available_gb"] == 18.0  # 24 - 6
            assert hardware["gpu_compute_capability"] == (8, 9)
            assert hardware["gpu_device_count"] == 1
            assert hardware["fastembed_providers"] == [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            assert hardware["cpu_cores"] == 16

    def test_detect_hardware_cuda_error(self):
        """Test hardware detection with CUDA runtime error."""
        with (
            patch("torch.cuda.is_available", side_effect=RuntimeError("CUDA error")),
            patch("os.cpu_count", return_value=4),
        ):
            hardware = detect_hardware()

            # Should fallback to safe defaults
            assert hardware["cuda_available"] is False
            assert hardware["cpu_cores"] == 4

    def test_detect_hardware_vram_error(self):
        """Test hardware detection with VRAM allocation error."""
        mock_props = Mock()
        mock_props.name = "Test GPU"
        mock_props.total_memory = 8 * HW_BYTES_TO_GB_FACTOR
        mock_props.major = 7
        mock_props.minor = 5

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
            patch(
                "torch.cuda.memory_allocated", side_effect=RuntimeError("Memory error")
            ),
            patch("os.cpu_count", return_value=8),
        ):
            hardware = detect_hardware()

            assert hardware["cuda_available"] is True
            assert hardware["vram_total_gb"] == 8.0
            assert hardware["vram_available_gb"] == 8.0  # Falls back to total

    def test_detect_hardware_no_cpu_count(self):
        """Test hardware detection when os.cpu_count() returns None."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("os.cpu_count", return_value=None),
        ):
            hardware = detect_hardware()

            assert hardware["cpu_cores"] == DEFAULT_CPU_CORES
            assert hardware["cpu_threads"] == DEFAULT_CPU_CORES

    def test_detect_hardware_os_error(self):
        """Test hardware detection with OS error."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch(
                "torch.cuda.get_device_properties", side_effect=OSError("System error")
            ),
            patch("os.cpu_count", return_value=8),
        ):
            hardware = detect_hardware()

            # Should still complete with defaults
            assert hardware["cpu_cores"] == 8
            assert hardware["cuda_available"] is True


@pytest.mark.unit
class TestOptimalProviders:
    """Test optimal execution provider selection."""

    def test_get_optimal_providers_force_cpu(self):
        """Test forcing CPU-only execution."""
        with patch("torch.cuda.is_available", return_value=True):
            providers = get_optimal_providers(force_cpu=True)
            assert providers == ["CPUExecutionProvider"]

    def test_get_optimal_providers_with_cuda(self):
        """Test provider selection with CUDA available."""
        with patch("torch.cuda.is_available", return_value=True):
            providers = get_optimal_providers(force_cpu=False)
            assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_get_optimal_providers_no_cuda(self):
        """Test provider selection without CUDA."""
        with patch("torch.cuda.is_available", return_value=False):
            providers = get_optimal_providers(force_cpu=False)
            assert providers == ["CPUExecutionProvider"]


@pytest.mark.unit
class TestRecommendedBatchSize:
    """Test recommended batch size calculation."""

    def test_batch_size_no_cuda(self):
        """Test batch size recommendation without CUDA."""
        with patch("torch.cuda.is_available", return_value=False):
            # Test different model types
            assert get_recommended_batch_size("embedding") == 16
            assert get_recommended_batch_size("llm") == 1
            assert get_recommended_batch_size("vision") == 4
            assert get_recommended_batch_size("unknown") == DEFAULT_BATCH_SIZE_FALLBACK

    def test_batch_size_high_end_gpu(self):
        """Test batch size for high-end GPU (>=16GB)."""
        mock_props = Mock()
        mock_props.total_memory = 24 * HW_BYTES_TO_GB_FACTOR

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            assert get_recommended_batch_size("embedding") == 128
            assert get_recommended_batch_size("llm") == 8
            assert get_recommended_batch_size("vision") == 32
            assert get_recommended_batch_size("unknown") == 64

    def test_batch_size_mid_range_gpu(self):
        """Test batch size for mid-range GPU (8-15GB)."""
        mock_props = Mock()
        mock_props.total_memory = 12 * HW_BYTES_TO_GB_FACTOR

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            assert get_recommended_batch_size("embedding") == 64
            assert get_recommended_batch_size("llm") == 4
            assert get_recommended_batch_size("vision") == 16
            assert get_recommended_batch_size("unknown") == 32

    def test_batch_size_entry_level_gpu(self):
        """Test batch size for entry-level GPU (4-7GB)."""
        mock_props = Mock()
        mock_props.total_memory = 6 * HW_BYTES_TO_GB_FACTOR

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            assert get_recommended_batch_size("embedding") == 32
            assert get_recommended_batch_size("llm") == 2
            assert get_recommended_batch_size("vision") == 8
            assert get_recommended_batch_size("unknown") == 16

    def test_batch_size_low_vram_gpu(self):
        """Test batch size for low VRAM GPU (<4GB)."""
        mock_props = Mock()
        mock_props.total_memory = 2 * HW_BYTES_TO_GB_FACTOR

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            assert get_recommended_batch_size("embedding") == 16
            assert get_recommended_batch_size("llm") == 1
            assert get_recommended_batch_size("vision") == 4
            assert get_recommended_batch_size("unknown") == 8

    def test_batch_size_cuda_error(self):
        """Test batch size fallback with CUDA error."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "torch.cuda.get_device_properties",
                side_effect=RuntimeError("CUDA error"),
            ),
        ):
            assert get_recommended_batch_size("embedding") == 16  # CPU fallback
            assert get_recommended_batch_size("llm") == 1
            assert get_recommended_batch_size("unknown") == DEFAULT_BATCH_SIZE_FALLBACK

    def test_batch_size_os_error(self):
        """Test batch size fallback with OS error."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "torch.cuda.get_device_properties", side_effect=OSError("System error")
            ),
        ):
            assert get_recommended_batch_size("embedding") == 16  # CPU fallback

    def test_batch_size_value_error(self):
        """Test batch size fallback with value error."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "torch.cuda.get_device_properties",
                side_effect=ValueError("Invalid value"),
            ),
        ):
            assert get_recommended_batch_size("embedding") == 16  # CPU fallback


@pytest.mark.unit
class TestSpacyManager:
    """Test SpaCy manager functionality."""

    def test_spacy_manager_initialization(self):
        """Test SpaCy manager initialization."""
        manager = SpacyManager()
        # Basic sanity: can call ensure_model and receive an object (when installed)
        assert hasattr(manager, "ensure_model")

    def test_ensure_model_cached(self):
        """Test model loading with caching."""
        manager = SpacyManager()

        # Mock spaCy functions
        mock_nlp = Mock()
        mock_nlp.memory_zone = Mock()

        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("spacy.load", return_value=mock_nlp) as mock_load,
        ):
            # First call
            result1 = manager.ensure_model("en_core_web_sm")
            assert result1 == mock_nlp
            assert mock_load.call_count == 1

            # Second call should use cache (no extra spacy.load invocation)
            result2 = manager.ensure_model("en_core_web_sm")
            assert result2 == mock_nlp
            assert mock_load.call_count == 1  # No additional call

    def test_ensure_model_download_required(self):
        """Test model downloading when not installed."""
        manager = SpacyManager()
        mock_nlp = Mock()

        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=False
            ),
            patch("src.core.infrastructure.spacy_manager.download") as mock_download,
            patch("spacy.load", return_value=mock_nlp),
        ):
            result = manager.ensure_model("en_core_web_md")

            assert result == mock_nlp
            mock_download.assert_called_once_with("en_core_web_md")

    def test_ensure_model_thread_safety(self):
        """Test thread safety of model loading."""
        manager = SpacyManager()
        mock_nlp = Mock()
        load_calls = []

        def mock_load(model_name):
            # Simulate some processing time
            time.sleep(0.1)
            load_calls.append(model_name)
            return mock_nlp

        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("spacy.load", side_effect=mock_load),
        ):
            # Start multiple threads trying to load the same model
            threads = []
            results = []

            def load_model():
                result = manager.ensure_model("en_core_web_sm")
                results.append(result)

            for _ in range(5):
                thread = threading.Thread(target=load_model)
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # Should only load once despite multiple threads
            assert len(load_calls) == 1
            assert all(result == mock_nlp for result in results)

    def test_memory_optimized_processing(self):
        """Test memory-optimized processing context manager."""
        manager = SpacyManager()
        mock_nlp = Mock()
        mock_memory_zone = Mock()
        # Make the memory_zone usable as a context manager
        mock_memory_zone.__enter__ = Mock(return_value=None)
        mock_memory_zone.__exit__ = Mock(return_value=None)
        mock_nlp.memory_zone.return_value = mock_memory_zone

        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("spacy.load", return_value=mock_nlp),
            manager.memory_optimized_processing("en_core_web_sm") as nlp,
        ):
            assert nlp == mock_nlp
            mock_nlp.memory_zone.assert_called_once()

    def test_memory_optimized_processing_default_model(self):
        """Test memory-optimized processing with default model."""
        manager = SpacyManager()
        mock_nlp = Mock()
        mock_memory_zone = Mock()
        # Make the memory_zone usable as a context manager
        mock_memory_zone.__enter__ = Mock(return_value=None)
        mock_memory_zone.__exit__ = Mock(return_value=None)
        mock_nlp.memory_zone.return_value = mock_memory_zone

        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("spacy.load", return_value=mock_nlp) as mock_load,
        ):
            with manager.memory_optimized_processing() as nlp:
                assert nlp == mock_nlp

            # Should use default model name
            mock_load.assert_called_with("en_core_web_sm")

    def test_get_spacy_manager_singleton(self):
        """Test that get_spacy_manager returns a singleton."""
        manager1 = get_spacy_manager()
        manager2 = get_spacy_manager()

        assert manager1 is manager2
        assert isinstance(manager1, SpacyManager)

    def test_spacy_manager_multiple_models(self):
        """Test loading and caching multiple models."""
        manager = SpacyManager()
        mock_nlp_sm = Mock()
        mock_nlp_md = Mock()

        def mock_load(model_name):
            if model_name == "en_core_web_sm":
                return mock_nlp_sm
            elif model_name == "en_core_web_md":
                return mock_nlp_md
            return Mock()

        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("spacy.load", side_effect=mock_load),
        ):
            # Load two different models
            result_sm = manager.ensure_model("en_core_web_sm")
            result_md = manager.ensure_model("en_core_web_md")

            assert result_sm == mock_nlp_sm
            assert result_md == mock_nlp_md

    def test_spacy_manager_download_error(self):
        """Test error handling during model download."""
        manager = SpacyManager()

        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=False
            ),
            patch(
                "src.core.infrastructure.spacy_manager.download",
                side_effect=Exception("Download failed"),
            ),
            pytest.raises(Exception, match="Download failed"),
        ):
            manager.ensure_model("nonexistent_model")


@pytest.mark.unit
class TestHardwareUtilsConstants:
    """Test hardware utilities constants."""

    def test_hardware_constants_values(self):
        """Test that hardware constants have expected values."""
        assert HW_BYTES_TO_GB_FACTOR == 1024**3
        assert DEFAULT_CPU_CORES == 1
        assert DEFAULT_BATCH_SIZE_FALLBACK == 8
        assert HIGH_END_VRAM_THRESHOLD == 16
        assert MID_RANGE_VRAM_THRESHOLD == 8
        assert ENTRY_LEVEL_VRAM_THRESHOLD == 4

    def test_batch_size_constants_structure(self):
        """Test batch size constant dictionaries structure."""
        from src.core.infrastructure.hardware_utils import (
            CPU_BATCH_SIZES,
            ENTRY_LEVEL_BATCH_SIZES,
            HIGH_END_BATCH_SIZES,
            LOW_VRAM_BATCH_SIZES,
            MID_RANGE_BATCH_SIZES,
        )

        # All batch size dictionaries should have same keys
        expected_keys = {"embedding", "llm", "vision"}

        assert set(CPU_BATCH_SIZES.keys()) == expected_keys
        assert set(HIGH_END_BATCH_SIZES.keys()) == expected_keys
        assert set(MID_RANGE_BATCH_SIZES.keys()) == expected_keys
        assert set(ENTRY_LEVEL_BATCH_SIZES.keys()) == expected_keys
        assert set(LOW_VRAM_BATCH_SIZES.keys()) == expected_keys

        # Verify reasonable batch size ordering (high-end > mid > entry > low)
        assert HIGH_END_BATCH_SIZES["embedding"] > MID_RANGE_BATCH_SIZES["embedding"]
        assert MID_RANGE_BATCH_SIZES["embedding"] > ENTRY_LEVEL_BATCH_SIZES["embedding"]
        assert ENTRY_LEVEL_BATCH_SIZES["embedding"] > LOW_VRAM_BATCH_SIZES["embedding"]


@pytest.mark.integration
class TestCoreInfrastructureIntegration:
    """Integration tests for core infrastructure components."""

    def test_hardware_detection_integration(self):
        """Integration test for hardware detection with realistic scenarios."""
        # Test that hardware detection completes without errors
        hardware = detect_hardware()

        # Basic structure validation
        required_keys = {
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

        assert set(hardware.keys()) == required_keys
        assert isinstance(hardware["cuda_available"], bool)
        assert isinstance(hardware["gpu_device_count"], int)
        assert isinstance(hardware["fastembed_providers"], list)
        assert hardware["cpu_cores"] >= 1

    def test_batch_size_recommendation_consistency(self):
        """Test that batch size recommendations are consistent and reasonable."""
        model_types = ["embedding", "llm", "vision", "unknown"]

        for model_type in model_types:
            batch_size = get_recommended_batch_size(model_type)
            assert isinstance(batch_size, int)
            assert batch_size > 0
            assert batch_size <= 256  # Reasonable upper bound

    @pytest.mark.asyncio
    async def test_gpu_monitoring_lifecycle(self):
        """Test complete GPU monitoring lifecycle."""
        # Test that GPU monitoring can be called multiple times
        async with gpu_performance_monitor() as metrics1:
            # Metrics can be None (no CUDA) or GPUMetrics instance
            assert metrics1 is None or isinstance(metrics1, GPUMetrics)

        async with gpu_performance_monitor() as metrics2:
            assert metrics2 is None or isinstance(metrics2, GPUMetrics)

        # If both are not None, they should have same device info structure
        if metrics1 is not None and metrics2 is not None:
            assert isinstance(metrics1.device_name, str)
            assert isinstance(metrics2.device_name, str)
            assert metrics1.device_name == metrics2.device_name
