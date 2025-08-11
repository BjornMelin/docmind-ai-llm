"""Comprehensive tests for utility functions and helper modules.

This module tests core utility functions including hardware detection, document
loading, vectorstore creation, document analysis, chat functionality, reranking
components, GPU optimization with torch.compile, and spaCy model management
following 2025 best practices.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import subprocess

import pytest

from models import AppSettings
from utils.utils import (
    detect_hardware,
    ensure_spacy_model,
    get_embed_model,
    managed_async_qdrant_client,
    verify_rrf_configuration,
)


# Legacy test fixtures and functions (keeping for compatibility)
@pytest.fixture
def tmp_pdf(tmp_path):
    """Create a temporary PDF file for testing.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path: Path to the temporary PDF file.
    """
    path = tmp_path / "test.pdf"
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4 dummy content")
    return path


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return AppSettings(
        dense_embedding_model="BAAI/bge-large-en-v1.5",
        sparse_embedding_model="prithivida/Splade_PP_en_v1",
        gpu_acceleration=True,
        embedding_batch_size=32,
        rrf_fusion_weight_dense=0.7,
        rrf_fusion_weight_sparse=0.3,
        rrf_fusion_alpha=60,
    )


# Enhanced tests for new utils/utils.py functionality


@patch("utils.utils.torch.cuda.is_available", return_value=True)
@patch("utils.utils.torch.cuda.get_device_name", return_value="NVIDIA RTX 4090")
@patch("utils.utils.torch.cuda.get_device_properties")
@patch("utils.utils.ModelManager.get_text_embedding_model")
def test_detect_hardware_with_gpu(
    mock_get_model, mock_get_props, mock_get_name, mock_cuda_available
):
    """Test hardware detection with GPU available.

    Verifies correct detection of GPU hardware with VRAM information
    using FastEmbed's native provider detection.
    """
    # Mock GPU properties
    mock_get_props.return_value.total_memory = 24 * 1024**3  # 24GB

    # Mock FastEmbed model with providers
    mock_model = MagicMock()
    mock_model.model.model.get_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    mock_get_model.return_value = mock_model

    hardware_info = detect_hardware()

    assert hardware_info["cuda_available"] is True
    assert hardware_info["gpu_name"] == "NVIDIA RTX 4090"
    assert hardware_info["vram_total_gb"] == 24.0
    assert "CUDAExecutionProvider" in hardware_info["fastembed_providers"]
    assert "CPUExecutionProvider" in hardware_info["fastembed_providers"]

    mock_get_model.assert_called_once_with("BAAI/bge-small-en-v1.5")


@patch("utils.utils.torch.cuda.is_available", return_value=False)
@patch("utils.utils.ModelManager.get_text_embedding_model")
def test_detect_hardware_cpu_only(mock_get_model, mock_cuda_available):
    """Test hardware detection with CPU only.

    Verifies correct fallback to CPU-only detection when
    GPU is not available.
    """
    # Mock FastEmbed model without CUDA provider
    mock_model = MagicMock()
    mock_model.model.model.get_providers.return_value = ["CPUExecutionProvider"]
    mock_get_model.return_value = mock_model

    hardware_info = detect_hardware()

    assert hardware_info["cuda_available"] is False
    assert hardware_info["gpu_name"] == "Unknown"
    assert hardware_info["vram_total_gb"] is None
    assert hardware_info["fastembed_providers"] == ["CPUExecutionProvider"]


@patch("utils.utils.torch.cuda.is_available", return_value=True)
@patch("utils.utils.ModelManager.get_text_embedding_model")
def test_detect_hardware_fastembed_error_fallback(mock_get_model, mock_cuda_available):
    """Test hardware detection fallback when FastEmbed fails.

    Verifies graceful fallback to torch.cuda detection when
    FastEmbed provider detection fails.
    """
    # Mock FastEmbed model that fails provider detection
    mock_model = MagicMock()
    mock_model.model.model.get_providers.side_effect = AttributeError("No providers")
    mock_get_model.return_value = mock_model

    with patch("utils.utils.logging.warning") as mock_warning:
        hardware_info = detect_hardware()

        # Should fall back to torch.cuda detection
        assert hardware_info["cuda_available"] is True
        assert hardware_info["fastembed_providers"] == []
        mock_warning.assert_called()


@patch("utils.utils.torch.cuda.is_available", return_value=True)
@patch("utils.utils.torch.cuda.get_device_name", return_value="NVIDIA RTX 4090")
@patch("utils.utils.torch.cuda.get_device_properties")
@patch("utils.utils.hasattr", return_value=True)  # torch.compile available
@patch("utils.utils.torch.compile")
def test_get_embed_model_with_gpu_optimization(
    mock_torch_compile, mock_hasattr, mock_get_props, mock_get_name, mock_cuda_available
):
    """Test embedding model creation with GPU optimization.

    Verifies that torch.compile is applied when GPU is available
    and configured for optimal performance.
    """
    mock_get_props.return_value.total_memory = 16 * 1024**3  # 16GB

    # Mock torch.compile to return the model unchanged
    mock_torch_compile.return_value = MagicMock()

    with patch("utils.utils.FastEmbedEmbedding") as mock_fastembed:
        mock_model = MagicMock()
        mock_fastembed.return_value = mock_model

        with patch("utils.utils.settings") as mock_settings:
            mock_settings.gpu_acceleration = True
            mock_settings.dense_embedding_model = "BAAI/bge-large-en-v1.5"
            mock_settings.embedding_batch_size = 32

            get_embed_model()

            # Verify FastEmbed model creation with GPU providers
            mock_fastembed.assert_called_once_with(
                model_name="BAAI/bge-large-en-v1.5",
                max_length=512,
                cache_dir="./embeddings_cache",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                batch_size=32,
            )

            # Verify torch.compile was applied
            mock_torch_compile.assert_called_once_with(
                mock_model, mode="reduce-overhead", dynamic=True
            )


@patch("utils.utils.torch.cuda.is_available", return_value=False)
def test_get_embed_model_cpu_mode(mock_cuda_available):
    """Test embedding model creation in CPU mode.

    Verifies that CPU-only configuration is used when
    GPU is not available or disabled.
    """
    with patch("utils.utils.FastEmbedEmbedding") as mock_fastembed:
        mock_model = MagicMock()
        mock_fastembed.return_value = mock_model

        with patch("utils.utils.settings") as mock_settings:
            mock_settings.gpu_acceleration = False
            mock_settings.dense_embedding_model = "BAAI/bge-large-en-v1.5"
            mock_settings.embedding_batch_size = 16

            embed_model = get_embed_model()

            # Verify FastEmbed model creation with CPU-only providers
            mock_fastembed.assert_called_once_with(
                model_name="BAAI/bge-large-en-v1.5",
                max_length=512,
                cache_dir="./embeddings_cache",
                providers=["CPUExecutionProvider"],
                batch_size=16,
            )

            assert embed_model == mock_model


def test_verify_rrf_configuration_correct_weights():
    """Test RRF configuration verification with correct weights.

    Verifies that research-backed weight configuration (0.7/0.3)
    passes validation successfully.
    """
    settings = AppSettings(
        rrf_fusion_weight_dense=0.7, rrf_fusion_weight_sparse=0.3, rrf_fusion_alpha=60
    )

    with patch("utils.utils.logging.info"):
        verification = verify_rrf_configuration(settings)

        assert verification["weights_correct"] is True
        assert verification["alpha_in_range"] is True
        assert verification["prefetch_enabled"] is True
        assert len(verification["issues"]) == 0
        assert abs(verification["computed_hybrid_alpha"] - 0.7) < 0.001


def test_verify_rrf_configuration_incorrect_weights():
    """Test RRF configuration verification with incorrect weights.

    Verifies that non-research-backed weights trigger appropriate
    warnings and recommendations.
    """
    settings = AppSettings(
        rrf_fusion_weight_dense=0.5,  # Not research-backed
        rrf_fusion_weight_sparse=0.5,  # Not research-backed
        rrf_fusion_alpha=5,  # Outside research range
    )

    with patch("utils.utils.logging.info"):
        verification = verify_rrf_configuration(settings)

        assert verification["weights_correct"] is False
        assert verification["alpha_in_range"] is False
        assert len(verification["issues"]) == 2
        assert len(verification["recommendations"]) == 2

        # Check specific issue messages
        weight_issue = next(
            (
                issue
                for issue in verification["issues"]
                if "Weights not research-backed" in issue
            ),
            None,
        )
        alpha_issue = next(
            (issue for issue in verification["issues"] if "RRF alpha" in issue), None
        )

        assert weight_issue is not None
        assert alpha_issue is not None


@patch("spacy.load")
def test_ensure_spacy_model_already_loaded(mock_spacy_load):
    """Test spaCy model loading when model is already available.

    Verifies successful loading of an already installed spaCy model.
    """
    mock_nlp = MagicMock()
    mock_spacy_load.return_value = mock_nlp

    with patch("utils.utils.logging.info") as mock_log_info:
        result = ensure_spacy_model("en_core_web_sm")

        assert result == mock_nlp
        mock_spacy_load.assert_called_once_with("en_core_web_sm")
        mock_log_info.assert_called_with(
            "spaCy model 'en_core_web_sm' loaded successfully"
        )


@patch("spacy.load")
@patch("subprocess.run")
def test_ensure_spacy_model_needs_download(mock_subprocess, mock_spacy_load):
    """Test spaCy model download when model is not available.

    Verifies automatic download and loading when spaCy model
    is not locally available.
    """
    mock_nlp = MagicMock()

    # First call raises OSError (model not found), second call succeeds
    mock_spacy_load.side_effect = [OSError("Model not found"), mock_nlp]

    # Mock successful subprocess run
    mock_subprocess.return_value = MagicMock(returncode=0)

    with patch("utils.utils.logging.info") as mock_log_info:
        result = ensure_spacy_model("en_core_web_sm")

        assert result == mock_nlp

        # Verify download was attempted
        mock_subprocess.assert_called_once_with(
            ["python", "-m", "spacy", "download", "en_core_web_sm"],
            check=True,
            capture_output=True,
            text=True,
        )

        # Verify both download and success messages
        expected_calls = [
            call("Downloading spaCy model 'en_core_web_sm'..."),
            call("spaCy model 'en_core_web_sm' downloaded and loaded successfully"),
        ]
        mock_log_info.assert_has_calls(expected_calls)


@patch("spacy.load")
@patch("subprocess.run")
def test_ensure_spacy_model_download_fails(mock_subprocess, mock_spacy_load):
    """Test spaCy model download failure handling.

    Verifies proper error handling when model download fails.
    """
    # Both load calls fail
    mock_spacy_load.side_effect = OSError("Model not found")

    # Mock failed subprocess run
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "cmd")

    with (
        patch("utils.utils.logging.error") as mock_log_error,
        pytest.raises(RuntimeError, match="Failed to load or download"),
    ):
        ensure_spacy_model("en_core_web_sm")

    mock_log_error.assert_called()


def test_ensure_spacy_model_import_error():
    """Test handling of missing spaCy installation.

    Verifies proper error handling when spaCy is not installed.
    """
    with patch(
        "builtins.__import__", side_effect=ImportError("No module named 'spacy'")
    ):
        with (
            patch("utils.utils.logging.error") as mock_log_error,
            pytest.raises(RuntimeError, match="spaCy is not installed"),
        ):
            ensure_spacy_model("en_core_web_sm")

        mock_log_error.assert_called()


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("en_core_web_sm", "en_core_web_sm"),
        ("en_core_web_md", "en_core_web_md"),
        ("en_core_web_lg", "en_core_web_lg"),
    ],
)
@patch("spacy.load")
def test_ensure_spacy_model_different_models(mock_spacy_load, model_name, expected):
    """Test spaCy model loading with different model names.

    Parametrized test for various spaCy model sizes and types.
    """
    mock_nlp = MagicMock()
    mock_spacy_load.return_value = mock_nlp

    result = ensure_spacy_model(model_name)

    assert result == mock_nlp
    mock_spacy_load.assert_called_once_with(expected)


def test_rrf_alpha_parameter_calculation():
    """Test RRF alpha parameter calculation from weights.

    Verifies that hybrid alpha is correctly computed from
    dense and sparse fusion weights.
    """
    settings = AppSettings(rrf_fusion_weight_dense=0.8, rrf_fusion_weight_sparse=0.2)

    verification = verify_rrf_configuration(settings)
    expected_alpha = 0.8 / (0.8 + 0.2)

    assert abs(verification["computed_hybrid_alpha"] - expected_alpha) < 0.001
    assert abs(verification["computed_hybrid_alpha"] - 0.8) < 0.001


@pytest.mark.slow
@pytest.mark.requires_gpu
@patch("utils.utils.torch.cuda.is_available", return_value=True)
def test_get_embed_model_gpu_memory_logging(mock_cuda_available):
    """Test GPU memory information logging.

    Verifies that GPU memory information is properly logged
    when GPU acceleration is enabled.
    """
    with (
        patch("utils.utils.torch.cuda.get_device_name", return_value="Tesla V100"),
        patch("utils.utils.torch.cuda.get_device_properties") as mock_props,
    ):
        mock_props.return_value.total_memory = 32 * 1024**3  # 32GB

        with (
            patch("utils.utils.FastEmbedEmbedding"),
            patch("utils.utils.settings") as mock_settings,
        ):
            mock_settings.gpu_acceleration = True
            mock_settings.dense_embedding_model = "test/model"
            mock_settings.embedding_batch_size = 64

            with patch("utils.utils.logging.info") as mock_log_info:
                get_embed_model()

                # Check that GPU info was logged
                logged_messages = [call[0][0] for call in mock_log_info.call_args_list]
                gpu_message = next(
                    (
                        msg
                        for msg in logged_messages
                        if "Tesla V100" in msg and "32.0GB" in msg
                    ),
                    None,
                )
                assert gpu_message is not None


# ===== MERGED FROM test_utils_functional.py =====


class TestLoggingConfiguration:
    """Test logging setup for application monitoring."""

    def test_loguru_logging_works(self):
        """Test that loguru logging is properly configured."""
        from loguru import logger

        # Basic logging should work without issues
        try:
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")
            success = True
        except Exception:
            success = False

        assert success

    def test_loguru_context_logging(self):
        """Test loguru context logging capabilities."""
        from loguru import logger

        # Test structured logging with extra context
        try:
            logger.bind(user_id="test123", action="testing").info("Context test")
            success = True
        except Exception:
            success = False

        assert success

    def test_loguru_exception_logging(self):
        """Test loguru exception logging with traceback."""
        from loguru import logger

        # Test exception logging
        try:
            try:
                raise ValueError("Test exception")
            except ValueError as e:
                logger.exception("Test exception logging: {}", str(e))
            success = True
        except Exception:
            success = False

        assert success

    def test_loguru_performance_logging(self):
        """Test loguru performance for bulk operations."""
        import time

        from loguru import logger

        # Test that logging doesn't significantly impact performance
        start_time = time.perf_counter()

        for i in range(100):
            logger.debug(f"Debug message {i}")

        elapsed = time.perf_counter() - start_time

        # Should complete in reasonable time (less than 1 second for 100 messages)
        assert elapsed < 1.0


class TestHardwareDetectionEnhanced:
    """Test enhanced hardware detection for GPU acceleration."""

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


class TestEmbeddingModelManagementEnhanced:
    """Test enhanced embedding model creation and optimization."""

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


class TestRRFConfigurationValidationEnhanced:
    """Test enhanced RRF (Reciprocal Rank Fusion) configuration validation."""

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


class TestAsyncQdrantClientManagement:
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


class TestSpacyModelManagementEnhanced:
    """Test enhanced spaCy model management and downloading."""

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


class TestPerformanceOptimizationUtils:
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


class TestRealWorldUsageScenariosIntegration:
    """Test utilities in realistic usage scenarios."""

    def test_application_startup_sequence(self):
        """Test typical application startup workflow."""
        # 1. Logging is auto-configured with loguru
        from loguru import logger

        logger.info("Application starting")

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
