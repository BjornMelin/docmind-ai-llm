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

    # Core LLM Configuration (aligned with unified settings)
    assert settings.model_name == "Qwen/Qwen3-4B-Instruct-2507"
    assert settings.embedding_model == "BAAI/bge-large-en-v1.5"

    # Search and Retrieval
    assert settings.top_k == 10
    assert settings.rrf_fusion_weight_dense == 0.7

    # Hardware and Performance
    assert settings.enable_gpu_acceleration is True

    # Document Processing (aligned with unified settings)
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 50

    # Agent Configuration (updated property names)
    assert settings.max_agent_retries == 2
    assert settings.agent_decision_timeout == 300

    # Optimization
    assert settings.enable_document_caching is True

    # Infrastructure
    assert settings.vector_store_type == "qdrant"
    assert settings.use_reranking is True


@patch.dict(os.environ, {"DOCMIND_QDRANT_URL": "http://test:1234"})
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

    # Note: sparse embeddings are enabled by default in unified settings
    assert settings.use_sparse_embeddings is True


def test_rrf_fusion_weights():
    """Test RRF fusion weight configuration."""
    settings = DocMindSettings()

    assert settings.rrf_fusion_weight_dense == 0.7
    assert settings.rrf_fusion_weight_sparse == 0.3
    assert settings.rrf_fusion_alpha == 60

    # Test weight sum equals 1.0
    weight_sum = settings.rrf_fusion_weight_dense + settings.rrf_fusion_weight_sparse
    assert abs(weight_sum - 1.0) < 0.001


def test_rrf_fusion_alpha_validation():
    """Test RRF fusion alpha parameter validation."""
    # Test valid alpha values
    settings = DocMindSettings(rrf_fusion_alpha=60)
    assert settings.rrf_fusion_alpha == 60

    # Test boundary values
    DocMindSettings(rrf_fusion_alpha=10)  # Minimum
    DocMindSettings(rrf_fusion_alpha=100)  # Maximum

    # Test invalid values
    with pytest.raises(ValidationError):
        DocMindSettings(rrf_fusion_alpha=5)  # Too low

    with pytest.raises(ValidationError):
        DocMindSettings(rrf_fusion_alpha=150)  # Too high


def test_gpu_acceleration_settings():
    """Test GPU acceleration configuration settings."""
    settings = DocMindSettings()

    assert settings.enable_gpu_acceleration is True


def test_qdrant_url_configuration():
    """Test Qdrant URL configuration."""
    settings = DocMindSettings()

    assert settings.qdrant_url == "http://localhost:6333"  # Actual default value

    # Test environment override
    with patch.dict(os.environ, {"DOCMIND_QDRANT_URL": "http://qdrant:6333"}):
        settings = DocMindSettings()
        assert settings.qdrant_url == "http://qdrant:6333"


def test_embedding_dimension_validation():
    """Test embedding dimension validation."""
    # Test valid dimension
    settings = DocMindSettings(embedding_dimension=768)
    assert settings.embedding_dimension == 768

    # Test boundary values work (no specific validation implemented)
    DocMindSettings(embedding_dimension=256)  # Should work - minimum boundary
    DocMindSettings(embedding_dimension=4096)  # Should work - maximum boundary


def test_bge_model_dimension_compatibility():
    """Test BGE-Large model dimension compatibility."""
    # BGE-Large model with compatible dimensions should work
    settings = DocMindSettings(
        embedding_model="BAAI/bge-large-en-v1.5",
        embedding_dimension=1024,
    )
    assert settings.embedding_dimension == 1024


def test_environment_variable_loading():
    """Test environment variable loading for various settings."""
    test_cases = [
        ("DOCMIND_CHUNK_SIZE", "chunk_size", "512", 512),
        ("DOCMIND_CHUNK_OVERLAP", "chunk_overlap", "50", 50),
        ("DOCMIND_ENABLE_GPU_ACCELERATION", "enable_gpu_acceleration", "false", False),
        ("DOCMIND_USE_SPARSE_EMBEDDINGS", "use_sparse_embeddings", "true", True),
        ("DOCMIND_RRF_FUSION_ALPHA", "rrf_fusion_alpha", "45", 45),
    ]

    for env_var, field_name, env_value, expected_value in test_cases:
        with patch.dict(os.environ, {env_var: env_value}):
            settings = DocMindSettings()
            assert getattr(settings, field_name) == expected_value


def test_sparse_embeddings_configuration():
    """Test sparse embeddings configuration."""
    settings = DocMindSettings(use_sparse_embeddings=True)
    assert settings.use_sparse_embeddings is True

    # Sparse embeddings disabled
    settings = DocMindSettings(use_sparse_embeddings=False)
    assert settings.use_sparse_embeddings is False


def test_model_config_settings():
    """Test model configuration settings."""
    settings = DocMindSettings()

    # Verify configuration dict is set properly
    assert hasattr(settings, "model_config")
    assert settings.model_config.get("env_file") == ".env"
    assert settings.model_config.get("env_prefix") == "DOCMIND_"
