"""Clean test suite for Pydantic models that only tests existing fields.

This module tests the data models including AnalysisOutput and Settings configuration
with only the fields that actually exist in the current implementation.
"""

import json
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config.settings import DocMindSettings
from src.models.schemas import AnalysisOutput


def test_analysis_output_creation():
    """Test AnalysisOutput model creation with valid data."""
    output = AnalysisOutput(
        summary="Test summary",
        key_insights=["Insight 1", "Insight 2"],
        action_items=["Action 1", "Action 2"],
        open_questions=["Question 1", "Question 2"],
    )

    assert output.summary == "Test summary"
    assert len(output.key_insights) == 2
    assert len(output.action_items) == 2
    assert len(output.open_questions) == 2


def test_analysis_output_validation():
    """Test AnalysisOutput validation with invalid data."""
    with pytest.raises(ValidationError):
        AnalysisOutput(summary=123)  # Invalid type


def test_analysis_output_json_serialization():
    """Test AnalysisOutput JSON serialization and deserialization."""
    output = AnalysisOutput(
        summary="Test summary",
        key_insights=["Insight 1"],
        action_items=["Action 1"],
        open_questions=["Question 1"],
    )

    # Test serialization
    json_str = output.model_dump_json()
    data = json.loads(json_str)

    assert data["summary"] == "Test summary"
    assert data["key_insights"] == ["Insight 1"]

    # Test deserialization
    reconstructed = AnalysisOutput.model_validate(data)
    assert reconstructed.summary == output.summary
    assert reconstructed.key_insights == output.key_insights


def test_settings_default_values():
    """Test Settings model loads with expected default values."""
    settings = DocMindSettings()

    # Core LLM Configuration
    assert settings.model_name == "gpt-4"
    assert settings.embedding_model == "text-embedding-3-small"

    # Search and Retrieval
    assert settings.top_k == 10
    assert settings.rrf_fusion_weight_dense == 0.7

    # Hardware and Performance
    assert settings.enable_gpu_acceleration is True

    # Document Processing
    assert settings.chunk_size == 1024
    assert settings.chunk_overlap == 200

    # Reliability
    assert settings.max_retries == 3
    assert settings.timeout == 30

    # Optimization
    assert settings.enable_document_caching is True

    # Infrastructure
    assert settings.vector_store_type == "qdrant"
    assert settings.use_reranking is True


@patch.dict(os.environ, {"QDRANT_URL": "http://test:1234"})
def test_settings_environment_override():
    """Test Settings model respects environment variable overrides."""
    settings = DocMindSettings()
    assert settings.qdrant_url == "http://test:1234"


def test_dense_embedding_settings():
    """Test dense embedding configuration settings."""
    settings = DocMindSettings()

    assert settings.embedding_dimension == 1024
    assert settings.embedding_model == "BAAI/bge-large-en-v1.5"


def test_sparse_embedding_settings():
    """Test sparse embedding configuration settings."""
    settings = DocMindSettings()

    assert settings.sparse_embedding_model is None
    assert settings.use_sparse_embeddings is False


def test_rrf_fusion_weights():
    """Test RRF fusion weight configuration."""
    settings = DocMindSettings()

    assert settings.rrf_fusion_weight_dense == 0.7
    assert settings.rrf_fusion_weight_sparse == 0.3
    assert settings.rrf_fusion_alpha == 60

    # Test weight sum equals 1.0
    weight_sum = settings.rrf_fusion_weight_dense + settings.rrf_fusion_weight_sparse
    assert abs(weight_sum - 1.0) < 0.001


def test_rrf_fusion_weight_sum_validation():
    """Test RRF weights sum validation."""
    with pytest.raises(ValidationError, match="RRF weights must sum to 1.0"):
        DocMindSettings(rrf_fusion_weight_dense=0.8, rrf_fusion_weight_sparse=0.8)


def test_gpu_acceleration_settings():
    """Test GPU acceleration configuration settings."""
    settings = DocMindSettings()

    assert settings.enable_gpu_acceleration is True


def test_qdrant_url_configuration():
    """Test Qdrant URL configuration."""
    settings = DocMindSettings()

    assert settings.qdrant_url == "http://localhost:6333"  # Actual default value

    # Test environment override
    with patch.dict(os.environ, {"QDRANT_URL": "http://qdrant:6333"}):
        settings = DocMindSettings()
        assert settings.qdrant_url == "http://qdrant:6333"


def test_embedding_dimension_validation():
    """Test embedding dimension validation."""
    # Test valid dimension with non-BGE model
    settings = DocMindSettings(
        embedding_dimension=768,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    assert settings.embedding_dimension == 768

    # Test invalid dimension (too small)
    with pytest.raises(ValidationError, match="Embedding dimension must be positive"):
        DocMindSettings(embedding_dimension=0)

    # Test invalid dimension (too large)
    with pytest.raises(ValidationError, match="Embedding dimension seems too large"):
        DocMindSettings(embedding_dimension=20000)


def test_bge_model_dimension_validation():
    """Test BGE-Large model dimension validation."""
    # Should raise error if BGE-Large model doesn't have 1024 dimensions
    with pytest.raises(
        ValidationError, match="BGE-Large model requires 1024 dimensions"
    ):
        DocMindSettings(
            embedding_model="BAAI/bge-large-en-v1.5",
            embedding_dimension=768,
        )


def test_environment_variable_loading():
    """Test environment variable loading for various settings."""
    test_cases = [
        ("CHUNK_SIZE", "chunk_size", "512", 512),
        ("CHUNK_OVERLAP", "chunk_overlap", "100", 100),
        ("ENABLE_GPU_ACCELERATION", "enable_gpu_acceleration", "false", False),
        ("USE_SPARSE_EMBEDDINGS", "use_sparse_embeddings", "true", True),
        ("RRF_FUSION_ALPHA", "rrf_fusion_alpha", "45", 45),
    ]

    for env_var, field_name, env_value, expected_value in test_cases:
        with patch.dict(os.environ, {env_var: env_value}):
            settings = DocMindSettings()
            assert getattr(settings, field_name) == expected_value


def test_splade_model_name_validation():
    """Test SPLADE model name validation when sparse embeddings are enabled."""
    settings = DocMindSettings(
        use_sparse_embeddings=True,
        sparse_embedding_model="prithivida/Splade_PP_en_v1",
    )
    assert settings.sparse_embedding_model == "prithivida/Splade_PP_en_v1"


def test_model_config_settings():
    """Test model configuration settings."""
    settings = DocMindSettings()

    # Verify configuration dict is set properly
    assert "env_file" in settings.model_config
    assert settings.model_config["env_file"] == ".env"
    assert settings.model_config["env_file_encoding"] == "utf-8"
