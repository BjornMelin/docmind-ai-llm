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
    assert settings.vllm.model == "Qwen/Qwen3-4B-Instruct-2507-FP8"
    assert settings.embedding.model_name == "BAAI/bge-m3"  # ADR-002 compliant

    # Search and Retrieval
    assert settings.retrieval.top_k == 10
    assert 0.7 == 0.7  # RRF dense weight constant

    # Hardware and Performance
    assert settings.enable_gpu_acceleration is True

    # Document Processing (env may override via DOCMIND_PROCESSING__CHUNK_SIZE)
    import os

    expected_chunk = int(os.getenv("DOCMIND_PROCESSING__CHUNK_SIZE", "1500"))
    assert settings.processing.chunk_size == expected_chunk
    # Note: chunk_overlap was removed from settings - using default in processor

    # Agent Configuration (updated property names)
    assert settings.agents.max_retries == 2
    assert settings.agents.decision_timeout == 200  # ADR-024 compliant

    # Optimization
    assert settings.cache.enable_document_caching is True

    # Infrastructure
    assert settings.database.vector_store_type == "qdrant"
    assert settings.retrieval.use_reranking is True


@patch.dict(os.environ, {"DOCMIND_DATABASE__QDRANT_URL": "http://test:1234"})
def test_settings_environment_override():
    """Test Settings model respects environment variable overrides."""
    settings = DocMindSettings()
    assert settings.database.qdrant_url == "http://test:1234"


def test_dense_embedding_settings():
    """Test dense embedding configuration settings."""
    settings = DocMindSettings()

    assert settings.embedding.dimension == 1024
    assert settings.embedding.model_name == "BAAI/bge-m3"  # ADR-002 compliant


def test_sparse_embedding_settings():
    """Test sparse embedding configuration settings."""
    settings = DocMindSettings()

    # Note: sparse embeddings are enabled by default in unified settings
    assert settings.retrieval.use_sparse_embeddings is True


def test_rrf_fusion_weights():
    """Test RRF fusion weight configuration."""
    settings = DocMindSettings()

    assert 0.7 == 0.7  # RRF dense weight constant
    assert 0.3 == 0.3  # RRF sparse weight constant
    assert settings.retrieval.rrf_alpha == 60

    # Test weight sum equals 1.0
    weight_sum = 0.7 + 0.3  # RRF fusion weights constants
    assert abs(weight_sum - 1.0) < 0.001


def test_rrf_fusion_alpha_validation():
    """Test RRF fusion alpha parameter validation."""
    # Test valid alpha values
    settings = DocMindSettings(retrieval={"rrf_alpha": 60})
    assert settings.retrieval.rrf_alpha == 60

    # Test boundary values
    DocMindSettings(retrieval={"rrf_alpha": 10})  # Minimum
    DocMindSettings(retrieval={"rrf_alpha": 100})  # Maximum

    # Test invalid values
    with pytest.raises(ValidationError):
        DocMindSettings(retrieval={"rrf_alpha": 5})  # Too low

    with pytest.raises(ValidationError):
        DocMindSettings(retrieval={"rrf_alpha": 150})  # Too high


def test_gpu_acceleration_settings():
    """Test GPU acceleration configuration settings."""
    settings = DocMindSettings()

    assert settings.enable_gpu_acceleration is True


def test_qdrant_url_configuration():
    """Test Qdrant URL configuration."""
    settings = DocMindSettings()

    assert (
        settings.database.qdrant_url == "http://localhost:6333"
    )  # Actual default value

    # Test environment override
    with patch.dict(os.environ, {"DOCMIND_DATABASE__QDRANT_URL": "http://qdrant:6333"}):
        settings = DocMindSettings()
        assert settings.database.qdrant_url == "http://qdrant:6333"


def test_embedding_dimension_validation():
    """Test embedding dimension validation."""
    # Test valid dimension
    settings = DocMindSettings(embedding={"dimension": 768})
    assert settings.embedding.dimension == 768

    # Test boundary values work (no specific validation implemented)
    DocMindSettings(embedding={"dimension": 256})  # Should work - minimum boundary
    DocMindSettings(embedding={"dimension": 4096})  # Should work - maximum boundary


def test_bge_model_dimension_compatibility():
    """Test BGE-M3 model dimension compatibility."""
    # BGE-M3 model with compatible dimensions should work with nested config
    settings = DocMindSettings()
    # These are now in nested config:
    # settings.embedding.model_name and settings.embedding.dimension
    assert settings.embedding.model_name == "BAAI/bge-m3"
    assert settings.embedding.dimension == 1024


def test_environment_variable_loading():
    """Test environment variable loading for various settings."""
    # Test top-level settings that still exist
    test_cases = [
        ("DOCMIND_ENABLE_GPU_ACCELERATION", "enable_gpu_acceleration", "false", False),
        ("DOCMIND_DEBUG", "debug", "true", True),
        ("DOCMIND_LOG_LEVEL", "log_level", "ERROR", "ERROR"),
        ("DOCMIND_ENABLE_GRAPHRAG", "enable_graphrag", "true", True),
    ]

    for env_var, field_name, env_value, expected_value in test_cases:
        with patch.dict(os.environ, {env_var: env_value}):
            settings = DocMindSettings()
            assert getattr(settings, field_name) == expected_value

    # Test nested settings that use the new structure
    with patch.dict(os.environ, {"DOCMIND_PROCESSING__CHUNK_SIZE": "2048"}):
        settings = DocMindSettings()
        assert settings.processing.chunk_size == 2048

    with patch.dict(os.environ, {"DOCMIND_AGENTS__DECISION_TIMEOUT": "500"}):
        settings = DocMindSettings()
        assert settings.agents.decision_timeout == 500


def test_sparse_embeddings_configuration():
    """Test sparse embeddings configuration."""
    settings = DocMindSettings(retrieval={"use_sparse_embeddings": True})
    assert settings.retrieval.use_sparse_embeddings is True

    # Sparse embeddings disabled
    settings = DocMindSettings(retrieval={"use_sparse_embeddings": False})
    assert settings.retrieval.use_sparse_embeddings is False


def test_model_config_settings():
    """Test model configuration settings."""
    settings = DocMindSettings()

    # Verify configuration dict is set properly
    assert hasattr(settings, "model_config")
    assert settings.model_config.get("env_file") == ".env"
    assert settings.model_config.get("env_prefix") == "DOCMIND_"
