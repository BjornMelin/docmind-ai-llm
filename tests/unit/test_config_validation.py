"""Comprehensive tests for configuration validation fixes.

This module tests all the critical configuration validation fixes including
RRF weight validation, embedding dimension validation, model compatibility
checks, and startup configuration validation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models import AppSettings
from utils.utils import validate_startup_configuration


class TestRRFWeightValidation:
    """Test RRF weight validation functionality."""

    def test_rrf_weights_sum_validation_success(self):
        """Test RRF weights that correctly sum to 1.0."""
        settings = AppSettings(
            rrf_fusion_weight_dense=0.7, rrf_fusion_weight_sparse=0.3
        )
        assert settings.rrf_fusion_weight_dense == 0.7
        assert settings.rrf_fusion_weight_sparse == 0.3

    def test_rrf_weights_sum_validation_failure(self):
        """Test RRF weights must sum to 1.0."""
        with pytest.raises(ValidationError, match="RRF weights must sum to 1.0"):
            AppSettings(
                rrf_fusion_weight_dense=0.6,
                rrf_fusion_weight_sparse=0.3,  # Sum = 0.9, not 1.0
            )

    def test_rrf_weights_sum_validation_edge_case(self):
        """Test RRF weights with floating point precision."""
        with pytest.raises(ValidationError, match="RRF weights must sum to 1.0"):
            AppSettings(
                rrf_fusion_weight_dense=0.7001,  # Sum = 1.0001, outside tolerance
                rrf_fusion_weight_sparse=0.3,
            )

    def test_rrf_weights_range_validation_dense(self):
        """Test RRF dense weight must be in [0, 1] range."""
        with pytest.raises(ValidationError, match="RRF weight must be between 0 and 1"):
            AppSettings(
                rrf_fusion_weight_dense=1.5,  # Invalid: > 1
                rrf_fusion_weight_sparse=0.3,
            )

        with pytest.raises(ValidationError, match="RRF weight must be between 0 and 1"):
            AppSettings(
                rrf_fusion_weight_dense=-0.1,  # Invalid: < 0
                rrf_fusion_weight_sparse=0.3,
            )

    def test_rrf_weights_range_validation_sparse(self):
        """Test RRF sparse weight must be in [0, 1] range."""
        with pytest.raises(ValidationError, match="RRF weight must be between 0 and 1"):
            AppSettings(
                rrf_fusion_weight_dense=0.7,
                rrf_fusion_weight_sparse=1.2,  # Invalid: > 1
            )

        with pytest.raises(ValidationError, match="RRF weight must be between 0 and 1"):
            AppSettings(
                rrf_fusion_weight_dense=0.7,
                rrf_fusion_weight_sparse=-0.5,  # Invalid: < 0
            )

    def test_rrf_weights_boundary_values(self):
        """Test RRF weights at boundary values."""
        # Test boundary case: all weight on dense
        settings = AppSettings(
            rrf_fusion_weight_dense=1.0, rrf_fusion_weight_sparse=0.0
        )
        assert settings.rrf_fusion_weight_dense == 1.0
        assert settings.rrf_fusion_weight_sparse == 0.0

        # Test boundary case: all weight on sparse
        settings = AppSettings(
            rrf_fusion_weight_dense=0.0, rrf_fusion_weight_sparse=1.0
        )
        assert settings.rrf_fusion_weight_dense == 0.0
        assert settings.rrf_fusion_weight_sparse == 1.0

    def test_rrf_weights_custom_valid_combinations(self):
        """Test various valid RRF weight combinations."""
        valid_combinations = [(0.6, 0.4), (0.5, 0.5), (0.8, 0.2), (0.25, 0.75)]

        for dense, sparse in valid_combinations:
            settings = AppSettings(
                rrf_fusion_weight_dense=dense, rrf_fusion_weight_sparse=sparse
            )
            assert settings.rrf_fusion_weight_dense == dense
            assert settings.rrf_fusion_weight_sparse == sparse


class TestEmbeddingDimensionValidation:
    """Test embedding dimension validation functionality."""

    def test_embedding_dimension_validation_positive(self):
        """Test embedding dimension must be positive."""
        with pytest.raises(
            ValidationError, match="Embedding dimension must be positive"
        ):
            AppSettings(dense_embedding_dimension=-100)

        with pytest.raises(
            ValidationError, match="Embedding dimension must be positive"
        ):
            AppSettings(dense_embedding_dimension=0)

    def test_embedding_dimension_validation_reasonable_size(self):
        """Test embedding dimension must be reasonable size."""
        with pytest.raises(
            ValidationError, match="Embedding dimension seems too large"
        ):
            AppSettings(dense_embedding_dimension=15000)  # > 10000

    def test_embedding_dimension_validation_valid_sizes(self):
        """Test valid embedding dimension sizes."""
        valid_dimensions = [512, 768, 1024, 1536, 3072, 4096]

        for dim in valid_dimensions:
            settings = AppSettings(dense_embedding_dimension=dim)
            assert settings.dense_embedding_dimension == dim


class TestModelCompatibilityValidation:
    """Test model compatibility validation functionality."""

    def test_bge_large_dimension_validation_correct(self):
        """Test BGE-Large model with correct dimension."""
        settings = AppSettings(
            dense_embedding_model="BAAI/bge-large-en-v1.5",
            dense_embedding_dimension=1024,
        )
        assert settings.dense_embedding_model == "BAAI/bge-large-en-v1.5"
        assert settings.dense_embedding_dimension == 1024

    def test_bge_large_dimension_validation_incorrect(self):
        """Test BGE-Large model with incorrect dimension."""
        with pytest.raises(
            ValidationError, match="BGE-Large model requires 1024 dimensions"
        ):
            AppSettings(
                dense_embedding_model="BAAI/bge-large-en-v1.5",
                dense_embedding_dimension=768,  # Wrong dimension for BGE-Large
            )

    def test_splade_model_validation_correct(self):
        """Test SPLADE++ model with correct name."""
        settings = AppSettings(sparse_embedding_model="prithivida/Splade_PP_en_v1")
        assert settings.sparse_embedding_model == "prithivida/Splade_PP_en_v1"

    def test_splade_model_validation_incorrect(self):
        """Test SPLADE++ model with incorrect name."""
        with pytest.raises(ValidationError, match="Invalid SPLADE\\+\\+ model name"):
            AppSettings(sparse_embedding_model="wrong/splade-model")

    def test_splade_model_validation_none(self):
        """Test SPLADE++ model validation with None value."""
        # Should not validate when model is None or empty
        settings = AppSettings(sparse_embedding_model=None)
        assert settings.sparse_embedding_model is None


class TestChunkConfigurationValidation:
    """Test chunk configuration validation functionality."""

    def test_chunk_size_overlap_validation_valid(self):
        """Test valid chunk size and overlap configuration."""
        settings = AppSettings(chunk_size=1024, chunk_overlap=200)
        assert settings.chunk_size == 1024
        assert settings.chunk_overlap == 200

    def test_chunk_size_overlap_validation_invalid(self):
        """Test chunk size must be larger than overlap."""
        with pytest.raises(ValidationError, match="Chunk size .* must be larger"):
            AppSettings(
                chunk_size=500,
                chunk_overlap=600,  # Overlap > chunk_size
            )

        with pytest.raises(ValidationError, match="Chunk size .* must be larger"):
            AppSettings(
                chunk_size=1000,
                chunk_overlap=1000,  # Overlap == chunk_size
            )

    def test_chunk_size_overlap_boundary(self):
        """Test boundary case where chunk_size is just larger than overlap."""
        settings = AppSettings(
            chunk_size=501,
            chunk_overlap=500,  # Just valid
        )
        assert settings.chunk_size == 501
        assert settings.chunk_overlap == 500


class TestStartupConfigurationValidation:
    """Test startup configuration validation functionality."""

    @patch("qdrant_client.QdrantClient")
    def test_startup_validation_success(self, mock_qdrant_client):
        """Test successful startup configuration validation."""
        # Mock Qdrant client
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value = []

        settings = AppSettings()
        result = validate_startup_configuration(settings)

        assert result["valid"] is True
        assert len(result["info"]) > 0
        assert "Qdrant connection successful" in result["info"][0]

    @patch("qdrant_client.QdrantClient")
    def test_startup_validation_qdrant_failure(self, mock_qdrant_client):
        """Test startup validation with Qdrant connection failure."""
        # Mock Qdrant connection failure
        mock_qdrant_client.side_effect = Exception("Connection failed")

        settings = AppSettings()

        with pytest.raises(RuntimeError, match="Critical configuration errors"):
            validate_startup_configuration(settings)

    @patch("torch.cuda.is_available")
    @patch("qdrant_client.QdrantClient")
    def test_startup_validation_gpu_warning(
        self, mock_qdrant_client, mock_cuda_available
    ):
        """Test startup validation with GPU configuration warning."""
        # Mock successful Qdrant connection
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value = []

        # Mock no GPU available
        mock_cuda_available.return_value = False

        settings = AppSettings(gpu_acceleration=True)
        result = validate_startup_configuration(settings)

        assert result["valid"] is True
        assert any(
            "GPU acceleration enabled but no GPU available" in warning
            for warning in result["warnings"]
        )

    @patch("qdrant_client.QdrantClient")
    def test_startup_validation_embedding_dimension_warning(self, mock_qdrant_client):
        """Test startup validation with embedding dimension warning."""
        # Mock successful Qdrant connection
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value = []

        settings = AppSettings(
            dense_embedding_model="BAAI/bge-large-en-v1.5",
            dense_embedding_dimension=768,  # Wrong dimension triggers warning
        )

        # This should pass validation but generate warnings
        with pytest.raises(ValidationError):
            # The model validation should catch this first
            validate_startup_configuration(settings)

    @patch("qdrant_client.QdrantClient")
    def test_startup_validation_rrf_alpha_warning(self, mock_qdrant_client):
        """Test startup validation with RRF alpha warning."""
        # Mock successful Qdrant connection
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value = []

        settings = AppSettings(
            enable_sparse_embeddings=True,
            rrf_fusion_alpha=5,  # Outside typical range
        )
        result = validate_startup_configuration(settings)

        assert result["valid"] is True
        assert any(
            "RRF alpha" in warning and "outside typical range" in warning
            for warning in result["warnings"]
        )


class TestValidationIntegration:
    """Test integration of all validation features."""

    def test_multiple_validation_errors(self):
        """Test that multiple validation errors are caught."""
        with pytest.raises(ValidationError) as exc_info:
            AppSettings(
                rrf_fusion_weight_dense=0.6,  # Wrong sum
                rrf_fusion_weight_sparse=0.5,  # Wrong sum
                dense_embedding_dimension=-1,  # Invalid dimension
                chunk_size=100,  # Too small
                chunk_overlap=200,  # Larger than chunk_size
            )

        # Should catch multiple validation errors
        error_str = str(exc_info.value)
        assert "validation error" in error_str.lower()

    def test_default_configuration_valid(self):
        """Test that default configuration passes all validations."""
        settings = AppSettings()

        # Should not raise any validation errors
        assert settings.rrf_fusion_weight_dense == 0.7
        assert settings.rrf_fusion_weight_sparse == 0.3
        assert settings.dense_embedding_dimension == 1024
        assert settings.chunk_size > settings.chunk_overlap

    def test_environment_override_validation(self):
        """Test validation with environment variable overrides."""
        import os

        # Test invalid configuration via environment variables
        with (
            patch.dict(
                os.environ,
                {
                    "RRF_FUSION_WEIGHT_DENSE": "0.8",
                    "RRF_FUSION_WEIGHT_SPARSE": "0.3",  # Sum = 1.1, invalid
                },
            ),
            pytest.raises(ValidationError, match="RRF weights must sum to 1.0"),
        ):
            AppSettings()

    @patch("qdrant_client.QdrantClient")
    def test_full_startup_validation_integration(self, mock_qdrant_client):
        """Test complete startup validation with valid configuration."""
        # Mock successful Qdrant connection
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value = []

        # Create valid configuration
        settings = AppSettings(
            rrf_fusion_weight_dense=0.6,
            rrf_fusion_weight_sparse=0.4,
            dense_embedding_dimension=1024,
            chunk_size=1024,
            chunk_overlap=200,
        )

        result = validate_startup_configuration(settings)

        assert result["valid"] is True
        assert len(result["errors"]) == 0
