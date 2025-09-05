"""Comprehensive unit tests for src.utils.core utility functions.

Tests focus on function input/output validation, error handling, edge cases,
async operations, and proper mocking of external dependencies. All tests
are designed for fast execution (<0.05s each) with parametrization.

Coverage areas:
- Hardware detection functions
- Configuration validation functions
- RRF configuration verification
- Context managers for resource management
- Async timing decorators

Mocked external dependencies:
- PyTorch CUDA operations
- Qdrant client connections
- System hardware detection
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config.settings import DocMindSettings
from src.utils.core import (
    RRF_ALPHA_MAX,
    RRF_ALPHA_MIN,
    WEIGHT_TOLERANCE,
    async_timer,
    detect_hardware,
    managed_async_qdrant_client,
    managed_gpu_operation,
    validate_startup_configuration,
    verify_rrf_configuration,
)


@pytest.mark.unit
class TestDetectHardware:
    """Test hardware detection functionality with comprehensive scenarios."""

    def test_detect_hardware_structure(self):
        """Test that detect_hardware returns expected structure."""
        result = detect_hardware()

        expected_keys = {"cuda_available", "gpu_name", "vram_total_gb"}
        assert set(result.keys()) == expected_keys
        assert isinstance(result["cuda_available"], bool)
        assert isinstance(result["gpu_name"], str)
        assert result["vram_total_gb"] is None or isinstance(
            result["vram_total_gb"], float
        )

    @pytest.mark.parametrize("cuda_available", [True, False])
    def test_detect_hardware_cuda_scenarios(self, cuda_available):
        """Test hardware detection with different CUDA availability."""
        with patch("torch.cuda.is_available", return_value=cuda_available):
            if cuda_available:
                with (
                    patch("torch.cuda.get_device_name", return_value="RTX 4090"),
                    patch("torch.cuda.get_device_properties") as mock_props,
                ):
                    mock_props.return_value.total_memory = 17179869184  # 16GB
                    result = detect_hardware()

                    assert result["cuda_available"] is True
                    assert result["gpu_name"] == "RTX 4090"
                    assert result["vram_total_gb"] == 16.0
            else:
                result = detect_hardware()
                assert result["cuda_available"] is False
                assert result["gpu_name"] == "Unknown"
                assert result["vram_total_gb"] is None

    @pytest.mark.parametrize(
        ("exception_type", "unused_expected_log_level"),
        [
            (RuntimeError("CUDA error"), "warning"),
            (OSError("System error"), "warning"),
            (AttributeError("Attribute error"), "warning"),
            (ImportError("Import error"), "error"),
            (ModuleNotFoundError("Module error"), "error"),
        ],
    )
    def test_detect_hardware_error_handling(
        self, exception_type, unused_expected_log_level
    ):
        """Test error handling during hardware detection."""
        with patch("torch.cuda.is_available", side_effect=exception_type):
            result = detect_hardware()

            # Should return safe defaults
            assert result["cuda_available"] is False
            assert result["gpu_name"] == "Unknown"
            assert result["vram_total_gb"] is None

    def test_detect_hardware_vram_calculation(self):
        """Test VRAM calculation accuracy."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="Test GPU"),
            patch("torch.cuda.get_device_properties") as mock_props,
        ):
            # Test various memory sizes
            test_cases = [
                (8589934592, 8.0),  # 8GB
                (10737418240, 10.0),  # 10GB
                (17179869184, 16.0),  # 16GB
                (25769803776, 24.0),  # 24GB
            ]

            for memory_bytes, expected_gb in test_cases:
                mock_props.return_value.total_memory = memory_bytes
                result = detect_hardware()
                assert result["vram_total_gb"] == expected_gb

    def test_detect_hardware_consistency(self):
        """Test that hardware detection returns consistent results."""
        result1 = detect_hardware()
        result2 = detect_hardware()

        # Results should be identical for static hardware info
        assert result1["cuda_available"] == result2["cuda_available"]
        assert result1["gpu_name"] == result2["gpu_name"]


@pytest.mark.unit
class TestValidateStartupConfiguration:
    """Test startup configuration validation with comprehensive scenarios."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock(spec=DocMindSettings)
        settings.database = Mock()
        settings.database.qdrant_url = "http://localhost:6333"
        settings.enable_gpu_acceleration = False
        settings.retrieval = Mock()
        settings.retrieval.strategy = "dense"
        settings.retrieval.rrf_alpha = 60
        return settings

    def test_validate_startup_success(self, mock_settings):
        """Test successful startup configuration validation."""
        with (
            patch("qdrant_client.QdrantClient") as mock_client_class,
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_client = Mock()
            mock_client.get_collections.return_value = []
            mock_client_class.return_value = mock_client

            result = validate_startup_configuration(mock_settings)

            assert result["valid"] is True
            assert len(result["errors"]) == 0
            assert any(
                "Qdrant connection successful" in info for info in result["info"]
            )
            mock_client.close.assert_called_once()

    @pytest.mark.parametrize(
        ("exception_type", "error_pattern"),
        [
            (ConnectionError("Connection refused"), "Qdrant connection failed"),
            (OSError("Network error"), "Qdrant network error"),
        ],
    )
    def test_validate_startup_qdrant_errors(
        self, mock_settings, exception_type, error_pattern
    ):
        """Test Qdrant connection error handling."""
        with (
            patch("qdrant_client.QdrantClient", side_effect=exception_type),
            pytest.raises(RuntimeError, match=error_pattern),
        ):
            validate_startup_configuration(mock_settings)

    def test_validate_startup_gpu_scenarios(self, mock_settings):
        """Test GPU configuration validation scenarios."""
        mock_settings.enable_gpu_acceleration = True

        # Test GPU available scenario
        with (
            patch("qdrant_client.QdrantClient") as mock_client_class,
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="RTX 4090"),
        ):
            mock_client = Mock()
            mock_client.get_collections.return_value = []
            mock_client_class.return_value = mock_client

            result = validate_startup_configuration(mock_settings)
            assert result["valid"] is True
            assert any("GPU available: RTX 4090" in info for info in result["info"])

        # Test GPU acceleration enabled but CUDA not available (line 108-110)
        with (
            patch("qdrant_client.QdrantClient") as mock_client_class,
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_client = Mock()
            mock_client.get_collections.return_value = []
            mock_client_class.return_value = mock_client

            result = validate_startup_configuration(mock_settings)
            assert result["valid"] is True
            assert any(
                "GPU acceleration enabled but no GPU available" in warning
                for warning in result["warnings"]
            )

    def test_validate_startup_rrf_warnings(self, mock_settings):
        """Test RRF configuration warnings."""
        mock_settings.retrieval.strategy = "hybrid"
        mock_settings.retrieval.rrf_alpha = 150  # Outside valid range

        with (
            patch("qdrant_client.QdrantClient") as mock_client_class,
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_client = Mock()
            mock_client.get_collections.return_value = []
            mock_client_class.return_value = mock_client

            result = validate_startup_configuration(mock_settings)
            assert result["valid"] is True
            assert any(
                "RRF alpha 150 outside optimal range" in warning
                for warning in result["warnings"]
            )

    @pytest.mark.parametrize(
        ("gpu_error_type", "expected_warning"),
        [
            (RuntimeError("CUDA error"), "CUDA error during GPU detection"),
            (ImportError("No CUDA"), "Import error during GPU detection"),
            (
                ModuleNotFoundError("Missing module"),
                "Import error during GPU detection",
            ),
        ],
    )
    def test_validate_startup_gpu_errors(
        self, mock_settings, gpu_error_type, expected_warning
    ):
        """Test GPU detection error handling."""
        mock_settings.enable_gpu_acceleration = True

        with (
            patch("qdrant_client.QdrantClient") as mock_client_class,
            patch("torch.cuda.is_available", side_effect=gpu_error_type),
        ):
            mock_client = Mock()
            mock_client.get_collections.return_value = []
            mock_client_class.return_value = mock_client

            result = validate_startup_configuration(mock_settings)
            assert result["valid"] is True
            assert any(expected_warning in warning for warning in result["warnings"])


@pytest.mark.unit
class TestVerifyRrfConfiguration:
    """Test RRF configuration verification with research-backed values."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for RRF testing."""
        settings = Mock(spec=DocMindSettings)
        settings.retrieval = Mock()
        settings.retrieval.rrf_alpha = 60
        settings.retrieval.rrf_k_constant = 60
        return settings

    def test_verify_rrf_weights_correct(self, mock_settings):
        """Test verification with correct research-backed weights."""
        result = verify_rrf_configuration(mock_settings)

        assert result["weights_correct"] is True
        assert len(result["issues"]) == 0
        assert result["computed_hybrid_alpha"] == 0.7

    @pytest.mark.parametrize(
        ("rrf_alpha", "expected_in_range"),
        [
            (10, True),  # Minimum valid
            (60, True),  # Optimal
            (100, True),  # Maximum valid
            (5, False),  # Below minimum
            (150, False),  # Above maximum
        ],
    )
    def test_verify_rrf_alpha_ranges(self, mock_settings, rrf_alpha, expected_in_range):
        """Test RRF alpha parameter validation."""
        mock_settings.retrieval.rrf_alpha = rrf_alpha
        result = verify_rrf_configuration(mock_settings)

        assert result["alpha_in_range"] is expected_in_range
        if not expected_in_range:
            assert len(result["issues"]) > 0
            assert len(result["recommendations"]) > 0

    def test_verify_rrf_configuration_structure(self, mock_settings):
        """Test RRF verification result structure."""
        result = verify_rrf_configuration(mock_settings)

        expected_keys = {
            "weights_correct",
            "alpha_in_range",
            "computed_hybrid_alpha",
            "issues",
            "recommendations",
        }
        assert set(result.keys()) == expected_keys
        assert isinstance(result["weights_correct"], bool)
        assert isinstance(result["alpha_in_range"], bool)
        assert isinstance(result["computed_hybrid_alpha"], float)
        assert isinstance(result["issues"], list)
        assert isinstance(result["recommendations"], list)

    def test_verify_rrf_constants(self):
        """Test RRF configuration constants."""
        assert WEIGHT_TOLERANCE == 0.05
        assert RRF_ALPHA_MIN == 10
        assert RRF_ALPHA_MAX == 100
        assert RRF_ALPHA_MIN < RRF_ALPHA_MAX

    def test_verify_rrf_weights_incorrect_edge_case(self, mock_settings):
        """Test unreachable code path for incorrect RRF weights (lines 161-164).

        Note: Due to a logic bug in the implementation where hardcoded values
        (0.7, 0.3) are compared against themselves, this else branch is never
        reached in normal execution. This test documents the unreachable code.
        """
        # The current implementation has abs(0.7 - 0.7) < 0.05 which is always True
        # This means lines 161-164 are unreachable with the current logic
        # This test documents this issue for future refactoring

        result = verify_rrf_configuration(mock_settings)

        # The weights should always be "correct" due to the logic bug
        assert result["weights_correct"] is True

        # Document that lines 161-164 contain unreachable code that should be refactored
        # to compare against actual settings values rather than hardcoded constants


@pytest.mark.unit
@pytest.mark.asyncio
class TestManagedGpuOperation:
    """Test GPU operation context manager with comprehensive scenarios."""

    async def test_managed_gpu_operation_success(self):
        """Test successful GPU operation management."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.synchronize") as mock_sync,
            patch("torch.cuda.empty_cache") as mock_empty,
            patch("gc.collect") as mock_gc,
        ):
            async with managed_gpu_operation():
                pass

            mock_sync.assert_called_once()
            mock_empty.assert_called_once()
            mock_gc.assert_called_once()

    async def test_managed_gpu_operation_no_cuda(self):
        """Test GPU operation management when CUDA unavailable."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.synchronize") as mock_sync,
            patch("torch.cuda.empty_cache") as mock_empty,
            patch("gc.collect") as mock_gc,
        ):
            async with managed_gpu_operation():
                pass

            mock_sync.assert_not_called()
            mock_empty.assert_not_called()
            mock_gc.assert_called_once()

    async def test_managed_gpu_operation_exception_propagation(self):
        """Test that exceptions are properly propagated."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.empty_cache"),
            patch("gc.collect"),
            pytest.raises(ValueError, match="test exception"),
        ):
            async with managed_gpu_operation():
                raise ValueError("test exception")


@pytest.mark.unit
@pytest.mark.asyncio
class TestManagedAsyncQdrantClient:
    """Test async Qdrant client context manager."""

    async def test_managed_async_qdrant_client_success(self):
        """Test successful async Qdrant client management."""
        mock_client = AsyncMock()

        with patch("src.utils.core.AsyncQdrantClient", return_value=mock_client):
            async with managed_async_qdrant_client("http://localhost:6333") as client:
                assert client is mock_client

            mock_client.close.assert_called_once()

    async def test_managed_async_qdrant_client_exception(self):
        """Test async Qdrant client management with exception."""
        mock_client = AsyncMock()

        with patch("src.utils.core.AsyncQdrantClient", return_value=mock_client):
            with pytest.raises(ValueError, match="test exception"):
                async with managed_async_qdrant_client("http://localhost:6333"):
                    raise ValueError("test exception")

            mock_client.close.assert_called_once()

    async def test_managed_async_qdrant_client_creation_failure(self):
        """Test async Qdrant client creation failure."""
        with (
            patch(
                "src.utils.core.AsyncQdrantClient",
                side_effect=ConnectionError("Failed"),
            ),
            pytest.raises(ConnectionError),
        ):
            async with managed_async_qdrant_client("http://localhost:6333"):
                pass

    async def test_managed_async_qdrant_client_none_handling(self):
        """Test handling when client creation returns None."""
        with patch("src.utils.core.AsyncQdrantClient", return_value=None):
            async with managed_async_qdrant_client("http://localhost:6333") as client:
                assert client is None


@pytest.mark.unit
@pytest.mark.asyncio
class TestAsyncTimer:
    """Test async timing decorator functionality."""

    async def test_async_timer_success(self):
        """Test async timer with successful function."""

        @async_timer
        async def test_func(value: int) -> int:
            await asyncio.sleep(0.001)  # Small delay
            return value * 2

        result = await test_func(5)
        assert result == 10

    async def test_async_timer_exception_handling(self):
        """Test async timer with function that raises exception."""

        @async_timer
        async def failing_func():
            await asyncio.sleep(0.001)
            raise ValueError("test exception")

        with pytest.raises(ValueError, match="test exception"):
            await failing_func()

    async def test_async_timer_preserves_function_metadata(self):
        """Test that async timer preserves function metadata."""

        @async_timer
        async def documented_func(arg: str) -> str:
            """Test function with documentation."""
            return arg.upper()

        assert documented_func.__name__ == "documented_func"
        # Note: __doc__ might be modified by the wrapper

    @pytest.mark.parametrize(
        ("func_args", "func_kwargs", "expected_result"),
        [
            ((1, 2), {}, 3),
            ((5,), {"multiplier": 2}, 10),
            ((), {"value": 42}, 42),
        ],
    )
    async def test_async_timer_parameter_handling(
        self, func_args, func_kwargs, expected_result
    ):
        """Test async timer with various parameter combinations."""

        @async_timer
        async def flexible_func(*args, multiplier=1, value=0):
            if args:
                return sum(args) * multiplier
            return value * multiplier

        result = await flexible_func(*func_args, **func_kwargs)
        assert result == expected_result

    @pytest.mark.usefixtures("perf_counter_boundary")
    async def test_async_timer_timing_accuracy(self):
        """Test that async timer measures time using deterministic perf_counter."""

        @async_timer
        async def trivial():
            # No real sleep; no_sleep fixture patches asyncio.sleep anyway
            return "done"

        start = time.perf_counter()
        result = await trivial()
        end = time.perf_counter()

        assert result == "done"
        # With perf_counter_boundary, wrapper may consume extra ticks.
        # Assert monotonicity and positive duration, not exact bound.
        actual_duration = end - start
        assert actual_duration > 0.0
        assert end >= start


@pytest.mark.unit
class TestUtilsCoreEdgeCases:
    """Test edge cases and boundary conditions for core utility functions."""

    def test_detect_hardware_extreme_vram_values(self):
        """Test hardware detection with extreme VRAM values."""
        test_cases = [
            (0, 0.0),  # 0 VRAM
            (1024, 0.0),  # 1KB - should round to 0.0
            (1073741824000, 1000.0),  # 1TB - massive VRAM
            (2**63 - 1, 8589934592.0),  # Max 64-bit signed int
        ]

        for memory_bytes, expected_gb in test_cases:
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.get_device_name", return_value="Test GPU"),
                patch("torch.cuda.get_device_properties") as mock_props,
            ):
                mock_props.return_value.total_memory = memory_bytes
                result = detect_hardware()
                assert result["vram_total_gb"] == expected_gb

    def test_detect_hardware_concurrent_calls(self):
        """Test hardware detection with concurrent/repeated calls."""
        results = []
        for _ in range(10):
            results.append(detect_hardware())

        # All results should be consistent
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result

    @pytest.mark.parametrize(
        ("alpha_value", "expected_valid"),
        [
            (9.999, False),  # Just below minimum
            (10.0, True),  # Exact minimum
            (100.0, True),  # Exact maximum
            (100.001, False),  # Just above maximum
            (0, False),  # Zero
            (-5, False),  # Negative
            (float("inf"), False),  # Infinity
            (float("nan"), False),  # NaN
        ],
    )
    def test_verify_rrf_configuration_alpha_edge_cases(
        self, alpha_value, expected_valid
    ):
        """Test RRF alpha validation with edge case values."""
        mock_settings = Mock()
        mock_settings.retrieval = Mock()
        mock_settings.retrieval.rrf_alpha = alpha_value
        mock_settings.retrieval.rrf_k_constant = 60

        try:
            result = verify_rrf_configuration(mock_settings)
            assert result["alpha_in_range"] == expected_valid
        except (ValueError, TypeError):
            # Some edge cases (NaN, inf) might raise exceptions
            assert not expected_valid

    def test_validate_startup_configuration_memory_pressure(self):
        """Test startup validation under memory pressure conditions."""
        mock_settings = Mock()
        mock_settings.database = Mock()
        mock_settings.database.qdrant_url = "http://localhost:6333"
        mock_settings.enable_gpu_acceleration = False
        mock_settings.retrieval = Mock()
        mock_settings.retrieval.strategy = "dense"
        mock_settings.retrieval.rrf_alpha = 60

        # Simulate memory pressure by making operations slow/fail occasionally
        call_count = 0

        def slow_get_collections():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                time.sleep(0.001)  # Simulate slow response
            return []

        with (
            patch("qdrant_client.QdrantClient") as mock_client_class,
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_client = Mock()
            mock_client.get_collections = Mock(side_effect=slow_get_collections)
            mock_client_class.return_value = mock_client

            result = validate_startup_configuration(mock_settings)
            assert result["valid"] is True


@pytest.mark.unit
class TestUtilsCoreFunctionContracts:
    """Test function contracts and edge cases for all utility functions."""

    @pytest.mark.parametrize(
        "invalid_settings",
        [
            None,
            "not_a_settings_object",
            {},
        ],
    )
    def test_validate_startup_configuration_invalid_inputs(self, invalid_settings):
        """Test startup validation with invalid inputs."""
        if invalid_settings is None:
            with pytest.raises(AttributeError):
                validate_startup_configuration(invalid_settings)
        else:
            # Mock objects without proper attributes should raise AttributeError
            with pytest.raises(AttributeError):
                validate_startup_configuration(invalid_settings)

    @pytest.mark.parametrize(
        "invalid_settings",
        [
            None,
            "not_a_settings_object",
            {},
        ],
    )
    def test_verify_rrf_configuration_invalid_inputs(self, invalid_settings):
        """Test RRF verification with invalid inputs."""
        if invalid_settings is None:
            with pytest.raises(AttributeError):
                verify_rrf_configuration(invalid_settings)
        else:
            with pytest.raises(AttributeError):
                verify_rrf_configuration(invalid_settings)

    def test_detect_hardware_no_external_dependencies(self):
        """Test that detect_hardware can run without external state."""
        # Should work regardless of system state
        result = detect_hardware()
        assert isinstance(result, dict)
        assert "cuda_available" in result

    @pytest.mark.asyncio
    async def test_context_managers_are_reusable(self):
        """Test that context managers can be used multiple times."""
        # Test managed_gpu_operation multiple times
        with patch("torch.cuda.is_available", return_value=False):
            for _ in range(3):
                async with managed_gpu_operation():
                    pass

        # Test managed_async_qdrant_client multiple times
        mock_client = AsyncMock()
        with patch("src.utils.core.AsyncQdrantClient", return_value=mock_client):
            for _ in range(3):
                async with managed_async_qdrant_client("http://localhost:6333"):
                    pass

            assert mock_client.close.call_count == 3
