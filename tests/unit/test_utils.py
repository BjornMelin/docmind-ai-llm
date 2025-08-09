"""Comprehensive tests for utility functions and helper modules.

This module tests core utility functions including hardware detection, document
loading, vectorstore creation, document analysis, chat functionality, reranking
components, GPU optimization with torch.compile, and spaCy model management
following 2025 best practices.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import logging
import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

from models import AppSettings
from utils.utils import (
    detect_hardware,
    ensure_spacy_model,
    get_embed_model,
    setup_logging,
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


def test_setup_logging():
    """Test logging configuration setup.

    Verifies that logging is properly configured with handlers
    and correct log levels.
    """
    with patch("logging.basicConfig") as mock_basic_config:
        setup_logging("DEBUG")

        # Verify basicConfig was called with correct parameters
        mock_basic_config.assert_called_once()
        call_kwargs = mock_basic_config.call_args[1]

        assert call_kwargs["level"] == logging.DEBUG
        assert "%(asctime)s" in call_kwargs["format"]
        assert len(call_kwargs["handlers"]) == 2  # StreamHandler and FileHandler


def test_setup_logging_default_level():
    """Test logging setup with default INFO level."""
    with patch("logging.basicConfig") as mock_basic_config:
        setup_logging()  # Should default to INFO

        call_kwargs = mock_basic_config.call_args[1]
        assert call_kwargs["level"] == logging.INFO


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
        ensure_spacy_model("en_core_web_sm")

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
        ensure_spacy_model("en_core_web_sm")

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

    ensure_spacy_model(model_name)

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
