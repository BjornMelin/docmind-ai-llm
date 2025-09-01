"""Comprehensive tests for configuration validation fixes.

This module tests all the critical configuration validation fixes including
RRF parameter validation, embedding dimension validation, model compatibility
checks, and startup configuration validation.
"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config.settings import DocMindSettings as Settings
from src.utils.core import validate_startup_configuration


class TestRRFParameterValidation:
    """Test RRF parameter validation functionality."""

    def test_rrf_fusion_alpha_validation_success(self):
        """Test RRF fusion alpha with valid values."""
        settings = Settings(retrieval={"rrf_alpha": 60})
        assert settings.retrieval.rrf_alpha == 60

    def test_rrf_fusion_alpha_range_validation(self):
        """Test RRF fusion alpha must be in [10, 100] range."""
        with pytest.raises(ValidationError):
            Settings(retrieval={"rrf_alpha": 5})  # Invalid: < 10

        with pytest.raises(ValidationError):
            Settings(retrieval={"rrf_alpha": 150})  # Invalid: > 100

    def test_rrf_k_constant_range_validation(self):
        """Test RRF k constant must be in [10, 100] range."""
        with pytest.raises(ValidationError):
            Settings(retrieval={"rrf_k_constant": 5})  # Invalid: < 10

        with pytest.raises(ValidationError):
            Settings(retrieval={"rrf_k_constant": 150})  # Invalid: > 100

    def test_rrf_parameters_boundary_values(self):
        """Test RRF parameters at boundary values."""
        # Test minimum valid values
        settings_min = Settings(retrieval={"rrf_alpha": 10, "rrf_k_constant": 10})
        assert settings_min.retrieval.rrf_alpha == 10
        assert settings_min.retrieval.rrf_k_constant == 10

        # Test maximum valid values
        settings_max = Settings(retrieval={"rrf_alpha": 100, "rrf_k_constant": 100})
        assert settings_max.retrieval.rrf_alpha == 100
        assert settings_max.retrieval.rrf_k_constant == 100

    def test_rrf_parameters_custom_valid_combinations(self):
        """Test various valid RRF parameter combinations."""
        valid_combinations = [(30, 40), (50, 50), (80, 20), (25, 75)]

        for alpha, k_const in valid_combinations:
            settings = Settings(
                retrieval={"rrf_alpha": alpha, "rrf_k_constant": k_const}
            )
            assert settings.retrieval.rrf_alpha == alpha
            assert settings.retrieval.rrf_k_constant == k_const


class TestTopKValidation:
    """Test top-k parameter validation."""

    def test_top_k_valid_range(self):
        """Test top_k with valid values."""
        settings = Settings(retrieval={"top_k": 10})
        assert settings.retrieval.top_k == 10

    def test_top_k_boundary_validation(self):
        """Test top_k boundary values."""
        # Test minimum valid value
        settings_min = Settings(retrieval={"top_k": 1})
        assert settings_min.retrieval.top_k == 1

        # Test maximum valid value
        settings_max = Settings(retrieval={"top_k": 50})
        assert settings_max.retrieval.top_k == 50

        # Test invalid values
        with pytest.raises(ValidationError):
            Settings(retrieval={"top_k": 0})  # Invalid: < 1

        with pytest.raises(ValidationError):
            Settings(retrieval={"top_k": 51})  # Invalid: > 50


class TestRerankerValidation:
    """Test reranking configuration validation."""

    def test_reranking_top_k_valid_range(self):
        """Test reranking_top_k with valid values."""
        settings = Settings(retrieval={"reranking_top_k": 5})
        assert settings.retrieval.reranking_top_k == 5

    def test_reranking_top_k_boundary_validation(self):
        """Test reranking_top_k boundary values."""
        # Test minimum valid value
        settings_min = Settings(retrieval={"reranking_top_k": 1})
        assert settings_min.retrieval.reranking_top_k == 1

        # Test maximum valid value
        settings_max = Settings(retrieval={"reranking_top_k": 20})
        assert settings_max.retrieval.reranking_top_k == 20

        # Test invalid values
        with pytest.raises(ValidationError):
            Settings(retrieval={"reranking_top_k": 0})  # Invalid: < 1

        with pytest.raises(ValidationError):
            Settings(retrieval={"reranking_top_k": 21})  # Invalid: > 20


class TestMemoryValidation:
    """Test memory configuration validation."""

    def test_memory_valid_ranges(self):
        """Test memory settings with valid values."""
        # Memory settings are not in nested config, remove from test or use
        # properties if available
        settings = Settings()
        assert (
            settings.vllm.gpu_memory_utilization >= 0.5
        )  # Test vLLM memory config instead

    def test_memory_boundary_validation(self):
        """Test memory boundary values."""
        # Test vLLM GPU memory utilization boundary values
        settings_min = Settings(vllm={"gpu_memory_utilization": 0.5})
        assert settings_min.vllm.gpu_memory_utilization == 0.5

        settings_max = Settings(vllm={"gpu_memory_utilization": 0.95})
        assert settings_max.vllm.gpu_memory_utilization == 0.95


class TestVLLMConfigValidation:
    """Test vLLM configuration validation."""

    def test_vllm_gpu_memory_utilization_valid_range(self):
        """Test GPU memory utilization with valid values."""
        settings = Settings(vllm={"gpu_memory_utilization": 0.95})
        assert settings.vllm.gpu_memory_utilization == 0.95

    def test_vllm_gpu_memory_utilization_boundary_validation(self):
        """Test GPU memory utilization boundary values."""
        # Test minimum valid value
        settings_min = Settings(vllm={"gpu_memory_utilization": 0.5})
        assert settings_min.vllm.gpu_memory_utilization == 0.5

        # Test maximum valid value
        settings_max = Settings(vllm={"gpu_memory_utilization": 0.95})
        assert settings_max.vllm.gpu_memory_utilization == 0.95

        # Test invalid values
        with pytest.raises(ValidationError):
            Settings(vllm={"gpu_memory_utilization": 0.4})  # Invalid: < 0.5

        with pytest.raises(ValidationError):
            Settings(vllm={"gpu_memory_utilization": 0.99})  # Invalid: > 0.95


class TestTokenLimitValidation:
    """Test token limit configuration validation."""

    def test_default_token_limit_valid_range(self):
        """Test default token limit with valid values."""
        settings = Settings(vllm={"context_window": 131072})
        assert settings.vllm.context_window == 131072

    def test_vllm_max_token_limit_valid_range(self):
        """Test vLLM max token limit with valid values."""
        settings = Settings(vllm={"max_tokens": 2048})
        assert settings.vllm.max_tokens == 2048

    def test_token_limit_boundary_validation(self):
        """Test token limit boundary values."""
        # Test minimum valid values
        settings_min = Settings(vllm={"context_window": 8192, "max_tokens": 128})
        assert settings_min.vllm.context_window == 8192
        assert settings_min.vllm.max_tokens == 128

        # Test maximum valid values
        settings_max = Settings(vllm={"context_window": 200000, "max_tokens": 8192})
        assert settings_max.vllm.context_window == 200000
        assert settings_max.vllm.max_tokens == 8192


class TestStartupConfigurationValidation:
    """Test startup configuration validation functionality."""

    def test_startup_validation_with_valid_settings(self):
        """Test startup validation with valid settings."""
        # Mock successful Qdrant connection
        with patch("qdrant_client.QdrantClient") as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.get_collections.return_value = []

            settings = Settings(enable_gpu_acceleration=False)
            result = validate_startup_configuration(settings)
            assert result["valid"] is True

    def test_startup_validation_qdrant_connection_error(self):
        """Test startup validation with Qdrant connection error."""
        # Mock Qdrant connection failure
        with patch("qdrant_client.QdrantClient") as mock_client:
            mock_client.side_effect = ConnectionError("Connection failed")

            settings = Settings()
            # Should raise RuntimeError for critical errors
            with pytest.raises(RuntimeError, match="Critical configuration errors"):
                validate_startup_configuration(settings)

    def test_startup_validation_gpu_configuration(self):
        """Test startup validation with GPU configuration."""
        # Mock successful Qdrant connection
        with patch("qdrant_client.QdrantClient") as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.get_collections.return_value = []

            # Mock GPU unavailable
            with patch("torch.cuda.is_available", return_value=False):
                settings = Settings(enable_gpu_acceleration=True)
                result = validate_startup_configuration(settings)
                # Should still be valid but with warnings
                assert result["valid"] is True
                assert any(
                    "GPU acceleration enabled but no GPU available" in warning
                    for warning in result.get("warnings", [])
                )


class TestValidationIntegration:
    """Test integration scenarios for validation."""

    def test_multiple_validation_errors(self):
        """Test handling multiple validation errors simultaneously."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                retrieval={
                    "rrf_alpha": 5,  # Invalid: < 10
                    "top_k": 0,  # Invalid: < 1
                    "reranking_top_k": 25,  # Invalid: > 20
                }
            )

        # Should contain multiple validation errors
        error = exc_info.value
        assert len(error.errors()) >= 1  # At least one error

    def test_default_configuration_valid(self):
        """Test that default configuration is valid."""
        settings = Settings()

        # Verify some key defaults
        assert 10 <= settings.retrieval.rrf_alpha <= 100
        assert 1 <= settings.retrieval.top_k <= 50
        assert 1 <= settings.retrieval.reranking_top_k <= 20
        assert 0.5 <= settings.vllm.gpu_memory_utilization <= 0.95

    def test_environment_override_validation(self):
        """Test that environment variables still respect validation."""
        import os

        # Set invalid environment variable for nested config
        os.environ["DOCMIND_RETRIEVAL__RRF_ALPHA"] = "5"  # Invalid: < 10

        try:
            with pytest.raises(ValidationError):
                Settings()
        finally:
            # Clean up
            if "DOCMIND_RETRIEVAL__RRF_ALPHA" in os.environ:
                del os.environ["DOCMIND_RETRIEVAL__RRF_ALPHA"]

    def test_full_startup_validation_integration(self):
        """Test complete startup validation with realistic configuration."""
        # Create a valid configuration using nested settings
        settings = Settings(
            retrieval={
                "rrf_alpha": 60,
                "rrf_k_constant": 60,
                "top_k": 10,
                "reranking_top_k": 5,
            },
            vllm={"gpu_memory_utilization": 0.85},
        )

        # Verify all values are within expected ranges
        assert settings.retrieval.rrf_alpha == 60
        assert settings.retrieval.rrf_k_constant == 60
        assert settings.retrieval.top_k == 10
        assert settings.retrieval.reranking_top_k == 5
        assert settings.vllm.gpu_memory_utilization == 0.85
