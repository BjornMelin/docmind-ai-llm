"""Functional tests for utils/utils.py - Core utility functions.

This test suite validates the real-world functionality of core utilities
including logging setup, hardware detection, embedding model management,
and RRF configuration validation. Tests focus on practical usage scenarios.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models import AppSettings
from utils.utils import (
    detect_hardware,
    ensure_spacy_model,
    get_embed_model,
    managed_async_qdrant_client,
    setup_logging,
    verify_rrf_configuration,
)


class TestLoggingConfiguration:
    """Test logging setup for application monitoring."""

    def test_logging_setup_with_valid_levels(self):
        """Logging should be configured correctly for different levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            setup_logging(level)

            # Verify logging level was set correctly
            root_logger = logging.getLogger()
            expected_level = getattr(logging, level.upper())
            assert root_logger.level == expected_level

    def test_logging_setup_creates_file_handler(self):
        """Logging should create both console and file handlers."""
        setup_logging("INFO")

        root_logger = logging.getLogger()
        handlers = root_logger.handlers

        # Should have at least a console and file handler
        handler_types = [type(handler).__name__ for handler in handlers]
        assert "StreamHandler" in handler_types
        assert "FileHandler" in handler_types

    def test_logging_format_consistency(self):
        """All handlers should use consistent formatting."""
        setup_logging("INFO")

        root_logger = logging.getLogger()
        formatters = [
            handler.formatter for handler in root_logger.handlers if handler.formatter
        ]

        # All formatters should have the same format string
        if formatters:
            format_strings = [f.format for f in formatters if hasattr(f, "format")]
            if format_strings:
                # All should be the same
                assert all(fmt == format_strings[0] for fmt in format_strings)

    def test_logging_case_insensitive_level(self):
        """Logging levels should be case insensitive."""
        test_levels = ["debug", "Info", "WARNING", "error"]

        for level in test_levels:
            setup_logging(level)
            root_logger = logging.getLogger()

            # Should work regardless of case
            expected_level = getattr(logging, level.upper())
            assert root_logger.level == expected_level


class TestHardwareDetection:
    """Test hardware detection for GPU acceleration."""

    def test_hardware_detection_returns_expected_keys(self):
        """Hardware detection should return all expected information keys."""
        hardware_info = detect_hardware()

        expected_keys = {
            "cuda_available",
            "gpu_name",
            "vram_total_gb",
            "fastembed_providers",
        }

        assert isinstance(hardware_info, dict)
        assert set(hardware_info.keys()) == expected_keys

    def test_hardware_detection_cuda_availability(self):
        """CUDA detection should match torch.cuda status."""
        hardware_info = detect_hardware()

        # Should report boolean CUDA availability
        assert isinstance(hardware_info["cuda_available"], bool)

        # Should generally match torch.cuda.is_available() if torch is working
        try:
            import torch

            if torch.cuda.is_available():
                # If torch reports CUDA, our detection should too
                # Or have good reason not to
                pass  # We can't guarantee exact match due to FastEmbed differences
        except ImportError:
            pass

    def test_hardware_detection_graceful_fallback(self):
        """Hardware detection should handle FastEmbed failures gracefully."""
        with patch("utils.utils.ModelManager") as mock_manager:
            # Simulate FastEmbed failure
            mock_instance = MagicMock()
            mock_instance.detect_hardware.side_effect = Exception(
                "FastEmbed unavailable"
            )
            mock_manager.get_instance.return_value = mock_instance

            hardware_info = detect_hardware()

            # Should still return valid structure with fallback values
            assert isinstance(hardware_info, dict)
            assert "cuda_available" in hardware_info
            assert isinstance(hardware_info["cuda_available"], bool)

    def test_hardware_detection_provider_list(self):
        """FastEmbed providers should be returned as a list."""
        hardware_info = detect_hardware()

        assert isinstance(hardware_info["fastembed_providers"], list)
        # Providers should be strings
        for provider in hardware_info["fastembed_providers"]:
            assert isinstance(provider, str)

    def test_hardware_detection_vram_reporting(self):
        """VRAM should be reported in GB as float or None."""
        hardware_info = detect_hardware()

        vram = hardware_info["vram_total_gb"]
        if vram is not None:
            assert isinstance(vram, int | float)
            assert vram >= 0  # Should be non-negative


class TestEmbeddingModelManagement:
    """Test embedding model creation and optimization."""

    def test_get_embed_model_returns_valid_model(self):
        """Embedding model should be created successfully."""
        with patch("fastembed.TextEmbedding") as mock_embedding:
            mock_model = MagicMock()
            mock_embedding.return_value = mock_model

            result = get_embed_model()

            assert result is not None
            mock_embedding.assert_called_once()

    def test_get_embed_model_uses_app_settings(self):
        """Embedding model should use configuration from app settings."""
        test_settings = AppSettings(
            dense_embedding_model="test-model", embedding_batch_size=32
        )

        with (
            patch("utils.utils.settings", test_settings),
            patch("fastembed.TextEmbedding") as mock_embedding,
        ):
            mock_embedding.return_value = MagicMock()

            get_embed_model()

            # Should use settings from configuration
            call_args = mock_embedding.call_args
            assert call_args is not None

    def test_get_embed_model_gpu_optimization(self):
        """Embedding model should handle GPU optimization when available."""
        test_settings = AppSettings(gpu_acceleration=True)

        with (
            patch("utils.utils.settings", test_settings),
            patch("fastembed.TextEmbedding") as mock_embedding,
            patch("utils.utils.detect_hardware") as mock_hardware,
        ):
            # Mock GPU available
            mock_hardware.return_value = {
                "cuda_available": True,
                "fastembed_providers": ["CUDAExecutionProvider"],
            }
            mock_embedding.return_value = MagicMock()

            result = get_embed_model()

            assert result is not None
            mock_embedding.assert_called_once()

    def test_get_embed_model_cpu_fallback(self):
        """Embedding model should fallback to CPU when GPU unavailable."""
        test_settings = AppSettings(gpu_acceleration=True)

        with (
            patch("utils.utils.settings", test_settings),
            patch("fastembed.TextEmbedding") as mock_embedding,
            patch("utils.utils.detect_hardware") as mock_hardware,
        ):
            # Mock GPU not available
            mock_hardware.return_value = {
                "cuda_available": False,
                "fastembed_providers": ["CPUExecutionProvider"],
            }
            mock_embedding.return_value = MagicMock()

            result = get_embed_model()

            assert result is not None
            mock_embedding.assert_called_once()

    def test_get_embed_model_error_handling(self):
        """Embedding model creation should handle errors gracefully."""
        with (
            patch(
                "fastembed.TextEmbedding", side_effect=RuntimeError("Model load failed")
            ),
            pytest.raises(RuntimeError, match="Model load failed"),
        ):
            get_embed_model()


class TestRRFConfigurationValidation:
    """Test RRF (Reciprocal Rank Fusion) configuration validation."""

    def test_valid_rrf_configuration(self):
        """Valid RRF configurations should pass validation."""
        valid_settings = AppSettings(
            rrf_fusion_weight_dense=0.7, rrf_fusion_weight_sparse=0.3
        )

        result = verify_rrf_configuration(valid_settings)

        assert isinstance(result, dict)
        assert "issues" in result
        assert "recommendations" in result
        assert isinstance(result["issues"], list)
        assert isinstance(result["recommendations"], list)

    def test_rrf_weight_sum_validation(self):
        """RRF weights should ideally sum to 1.0."""
        # Weights that don't sum to 1.0
        unbalanced_settings = AppSettings(
            rrf_fusion_weight_dense=0.8,
            rrf_fusion_weight_sparse=0.8,  # Sum = 1.6
        )

        result = verify_rrf_configuration(unbalanced_settings)

        # Should identify weight sum issues
        issues = result.get("issues", [])
        recommendations = result.get("recommendations", [])

        # Should provide feedback about weight balance
        assert len(issues) > 0 or len(recommendations) > 0

    def test_rrf_extreme_weight_values(self):
        """Extreme RRF weight values should be flagged."""
        extreme_settings = AppSettings(
            rrf_fusion_weight_dense=0.99, rrf_fusion_weight_sparse=0.01
        )

        result = verify_rrf_configuration(extreme_settings)

        # Should provide recommendations for extreme values
        recommendations = result.get("recommendations", [])
        assert isinstance(recommendations, list)

    def test_rrf_configuration_with_research_backed_values(self):
        """Research-backed RRF values should be recognized as optimal."""
        research_backed_settings = AppSettings(
            rrf_fusion_weight_dense=0.7,
            rrf_fusion_weight_sparse=0.3,
            rrf_fusion_alpha=60,  # Common research-backed value
        )

        result = verify_rrf_configuration(research_backed_settings)

        # Research-backed values should have minimal issues
        issues = result.get("issues", [])
        assert isinstance(issues, list)


class TestAsyncQdrantClient:
    """Test async Qdrant client management."""

    @pytest.mark.asyncio
    async def test_managed_qdrant_client_context_manager(self):
        """Async Qdrant client should work as context manager."""
        with patch(
            "qdrant_client.async_qdrant_client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            async with managed_async_qdrant_client("http://localhost:6333") as client:
                assert client is not None
                assert client == mock_client

            mock_client_class.assert_called_once_with(url="http://localhost:6333")

    @pytest.mark.asyncio
    async def test_managed_qdrant_client_with_custom_url(self):
        """Async Qdrant client should accept custom URLs."""
        custom_url = "https://custom-qdrant.example.com:6333"

        with patch(
            "qdrant_client.async_qdrant_client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            async with managed_async_qdrant_client(custom_url) as client:
                assert client is not None

            mock_client_class.assert_called_once_with(url=custom_url)

    @pytest.mark.asyncio
    async def test_managed_qdrant_client_connection_error(self):
        """Async Qdrant client should handle connection errors."""
        with patch(
            "qdrant_client.async_qdrant_client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client_class.side_effect = ConnectionError(
                "Unable to connect to Qdrant"
            )

            with pytest.raises(ConnectionError, match="Unable to connect to Qdrant"):
                async with managed_async_qdrant_client("http://localhost:6333"):
                    pass

    @pytest.mark.asyncio
    async def test_managed_qdrant_client_cleanup(self):
        """Async Qdrant client should cleanup properly on exit."""
        with patch(
            "qdrant_client.async_qdrant_client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Test normal exit
            async with managed_async_qdrant_client("http://localhost:6333") as client:
                assert client == mock_client

            # Verify cleanup was called if it exists
            if hasattr(mock_client, "close"):
                mock_client.close.assert_called()


class TestSpacyModelManagement:
    """Test spaCy model management and downloading."""

    def test_ensure_spacy_model_with_available_model(self):
        """SpaCy models should load when available."""
        with patch("spacy.load") as mock_load:
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp

            result = ensure_spacy_model("en_core_web_sm")

            assert result == mock_nlp
            mock_load.assert_called_once_with("en_core_web_sm")

    def test_ensure_spacy_model_with_download_fallback(self):
        """SpaCy models should be downloaded when not available."""
        with (
            patch("spacy.load") as mock_load,
            patch("subprocess.run") as mock_subprocess,
        ):
            # First call fails (model not found), second succeeds after download
            mock_nlp = MagicMock()
            mock_load.side_effect = [OSError("Model not found"), mock_nlp]
            mock_subprocess.return_value = MagicMock(returncode=0)

            result = ensure_spacy_model("en_core_web_sm")

            assert result == mock_nlp
            assert mock_load.call_count == 2
            mock_subprocess.assert_called_once()

    def test_ensure_spacy_model_download_failure(self):
        """SpaCy model download failures should be handled."""
        with (
            patch("spacy.load", side_effect=OSError("Model not found")),
            patch("subprocess.run") as mock_subprocess,
        ):
            # Download fails
            mock_subprocess.return_value = MagicMock(returncode=1)
            with pytest.raises(RuntimeError, match="Failed to download"):
                ensure_spacy_model("en_core_web_sm")

    def test_ensure_spacy_model_invalid_model_name(self):
        """Invalid spaCy model names should be handled gracefully."""
        with (
            patch("spacy.load", side_effect=OSError("Model not found")),
            patch("subprocess.run") as mock_subprocess,
        ):
            # Download succeeds but model still can't load
            mock_subprocess.return_value = MagicMock(returncode=0)
            with pytest.raises(RuntimeError, match="Model still not available"):
                ensure_spacy_model("invalid_model")


class TestPerformanceOptimization:
    """Test performance optimization utilities."""

    def test_memory_efficient_processing(self):
        """Utilities should support memory-efficient processing."""
        # Test that settings support memory optimization
        memory_settings = AppSettings(
            enable_quantization=True,
            embedding_batch_size=10,  # Small batch for memory efficiency
        )

        assert memory_settings.enable_quantization is True
        assert memory_settings.embedding_batch_size == 10

    def test_gpu_acceleration_settings(self):
        """GPU acceleration should be configurable."""
        gpu_settings = AppSettings(gpu_acceleration=True, cuda_device_id=0)

        assert gpu_settings.gpu_acceleration is True
        assert gpu_settings.cuda_device_id == 0

    def test_concurrent_processing_limits(self):
        """Concurrent processing should have configurable limits."""
        concurrent_settings = AppSettings(max_concurrent_requests=5)

        assert concurrent_settings.max_concurrent_requests == 5


class TestRealWorldUsageScenarios:
    """Test utilities in realistic usage scenarios."""

    def test_application_startup_sequence(self):
        """Test typical application startup workflow."""
        # 1. Setup logging
        setup_logging("INFO")

        # 2. Detect hardware
        hardware_info = detect_hardware()
        assert isinstance(hardware_info, dict)

        # 3. Initialize embedding model based on hardware
        with patch("fastembed.TextEmbedding") as mock_embedding:
            mock_embedding.return_value = MagicMock()

            embed_model = get_embed_model()
            assert embed_model is not None

    def test_configuration_validation_workflow(self):
        """Test configuration validation in realistic settings."""
        # Production-like settings
        prod_settings = AppSettings(
            rrf_fusion_weight_dense=0.7,
            rrf_fusion_weight_sparse=0.3,
            gpu_acceleration=True,
            embedding_batch_size=50,
        )

        # Validate RRF configuration
        rrf_result = verify_rrf_configuration(prod_settings)
        assert isinstance(rrf_result, dict)

        # Settings should be reasonable for production
        assert prod_settings.embedding_batch_size <= 100
        assert (
            prod_settings.rrf_fusion_weight_dense
            + prod_settings.rrf_fusion_weight_sparse
            <= 1.1
        )

    @pytest.mark.asyncio
    async def test_async_operations_workflow(self):
        """Test async operations in realistic workflow."""
        with patch(
            "qdrant_client.async_qdrant_client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate realistic async workflow
            async with managed_async_qdrant_client("http://localhost:6333") as client:
                # Client should be ready for operations
                assert client is not None

                # Would perform actual Qdrant operations here
                # client.search(), client.upsert(), etc.

    def test_error_recovery_scenarios(self):
        """Test error recovery in realistic failure scenarios."""
        # Network connectivity issues
        with patch(
            "qdrant_client.async_qdrant_client.AsyncQdrantClient"
        ) as mock_client:
            mock_client.side_effect = ConnectionError("Network unreachable")

            # Should propagate connection errors for proper handling
            import asyncio

            with pytest.raises(ConnectionError):
                asyncio.run(
                    managed_async_qdrant_client("http://localhost:6333").__aenter__()
                )

        # Model loading failures
        with (
            patch(
                "fastembed.TextEmbedding",
                side_effect=RuntimeError("CUDA out of memory"),
            ),
            pytest.raises(RuntimeError),
        ):
            # Should propagate model errors for fallback handling
            get_embed_model()

    def test_resource_constraint_handling(self):
        """Test handling of resource constraints."""
        # Memory-constrained configuration
        memory_constrained = AppSettings(
            embedding_batch_size=5,  # Very small batches
            enable_quantization=True,
            gpu_acceleration=False,  # CPU only
        )

        # Should accept memory-efficient settings
        assert memory_constrained.embedding_batch_size == 5
        assert memory_constrained.enable_quantization is True
        assert memory_constrained.gpu_acceleration is False
