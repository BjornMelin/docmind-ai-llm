"""Comprehensive tests for Pydantic models and application settings.

This module tests the data models including AnalysisOutput, AppSettings configuration,
validation behavior, environment variable overrides, and new features like SPLADE++
sparse embeddings, GPU acceleration settings, and multimodal processing configuration.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models import AnalysisOutput, AppSettings


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment fixture for isolated testing.

    Removes all environment variables that could affect AppSettings
    to ensure tests run in isolation with predictable defaults.
    """
    # Define env vars that could affect AppSettings
    env_vars_to_clear = [
        "BACKEND",
        "OLLAMA_BASE_URL",
        "LMSTUDIO_BASE_URL",
        "LLAMACPP_MODEL_PATH",
        "DEFAULT_MODEL",
        "CONTEXT_SIZE",
        "QDRANT_URL",
        "DENSE_EMBEDDING_MODEL",
        "DENSE_EMBEDDING_DIMENSION",
        "SPARSE_EMBEDDING_MODEL",
        "ENABLE_SPARSE_EMBEDDINGS",
        "RRF_FUSION_WEIGHT_DENSE",
        "RRF_FUSION_WEIGHT_SPARSE",
        "RRF_FUSION_ALPHA",
        "GPU_ACCELERATION",
        "CUDA_DEVICE_ID",
        "EMBEDDING_BATCH_SIZE",
        "PREFETCH_FACTOR",
        "ENABLE_QUANTIZATION",
        "QUANTIZATION_TYPE",
        "MAX_CONCURRENT_REQUESTS",
        "DEFAULT_RERANKER_MODEL",
        "ENABLE_COLBERT_RERANKING",
        "RERANKING_TOP_K",
        "PARSE_STRATEGY",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "MAX_ENTITIES",
        "DEBUG_MODE",
        "SIMILARITY_TOP_K",
        "RERANKER_MODEL",
        "DEFAULT_EMBEDDING_MODEL",
    ]

    for env_var in env_vars_to_clear:
        monkeypatch.delenv(env_var, raising=False)

    return monkeypatch


def test_analysis_output_creation():
    """Test AnalysisOutput model creation and field access.

    Tests that AnalysisOutput can be created with valid data and fields
    are properly accessible.
    """
    output = AnalysisOutput(
        summary="Sum",
        key_insights=["insight"],
        action_items=["action"],
        open_questions=["question"],
    )
    assert output.summary == "Sum"


def test_analysis_output_validation():
    """Test AnalysisOutput model validation with invalid data.

    Tests that ValidationError is raised when invalid data types
    are provided to model fields.
    """
    with pytest.raises(ValidationError):
        AnalysisOutput(summary=123)  # Invalid type


def test_settings_default_values():
    """Test Settings model loads with expected default values.

    Tests that Settings model initializes with correct default
    configuration values for the application.
    """
    settings = AppSettings()
    assert settings.llm_model == "gpt-4"
    assert settings.chunk_size == 1024
    assert settings.chunk_overlap == 200


@patch.dict(os.environ, {"QDRANT_URL": "http://test:1234"})
def test_settings_environment_override():
    """Test Settings model respects environment variable overrides.

    Tests that Settings model properly uses environment variables
    to override default configuration values.
    """
    settings = AppSettings()
    assert settings.qdrant_url == "http://test:1234"


def test_app_settings_document_processing_defaults():
    """Test Settings document processing configuration defaults.

    Tests that Settings model initializes with correct default
    values for document processing configuration.
    """
    settings = AppSettings()
    assert settings.chunk_size == 1024
    assert settings.chunk_overlap == 200
    assert settings.similarity_top_k == 10


def test_unstructured_import():
    """Verify Unstructured library is available.

    Tests that the Unstructured library and its core components
    can be imported successfully, ensuring multimodal parsing
    capabilities are available.
    """
    try:
        from unstructured.partition.auto import partition
        from unstructured.partition.pdf import partition_pdf

        assert callable(partition_pdf)
        assert callable(partition)
    except ImportError as e:
        pytest.fail(f"Unstructured import failed: {e}")


# Enhanced tests for new SPLADE++ and hybrid search settings


def test_sparse_embedding_settings():
    """Test SPLADE++ sparse embedding configuration settings.

    Verifies that sparse embedding settings are properly configured
    with research-backed SPLADE++ model and enable flags.
    """
    settings = AppSettings()

    # Test SPLADE++ model configuration (fixed typo)
    assert settings.sparse_embedding_model == "prithivida/Splade_PP_en_v1"
    assert settings.enable_sparse_embeddings is True

    # Test BGE-Large dense embedding configuration
    assert settings.dense_embedding_model == "BAAI/bge-large-en-v1.5"
    assert settings.dense_embedding_dimension == 1024


def test_rrf_fusion_weights():
    """Test RRF fusion weight configuration.

    Verifies that RRF fusion weights are set to research-backed
    optimal values (0.7 dense, 0.3 sparse).
    """
    settings = AppSettings()

    assert settings.rrf_fusion_weight_dense == 0.7
    assert settings.rrf_fusion_weight_sparse == 0.3
    assert settings.rrf_fusion_alpha == 60  # Qdrant research optimal

    # Test weight sum equals 1.0
    weight_sum = settings.rrf_fusion_weight_dense + settings.rrf_fusion_weight_sparse
    assert abs(weight_sum - 1.0) < 0.001


def test_gpu_acceleration_settings():
    """Test GPU acceleration configuration settings.

    Verifies GPU settings that are actually available.
    """
    settings = AppSettings()

    assert settings.gpu_acceleration is True
    assert settings.gpu_enabled is True


def test_reranking_configuration():
    """Test reranking configuration.

    Verifies reranking settings that are actually available.
    """
    settings = AppSettings()

    assert settings.rerank_enabled is True


def test_document_processing_configuration():
    """Test document processing settings.

    Verifies document processing configuration settings.
    """
    settings = AppSettings()

    assert settings.chunk_size == 1024  # Optimal for embeddings
    assert settings.chunk_overlap == 200


def test_default_model_configuration():
    """Test default model configuration.

    Verifies the default model setting.
    """
    settings = AppSettings()

    assert settings.llm_model == "gpt-4"


@patch.dict(
    os.environ,
    {
        "SPARSE_EMBEDDING_MODEL": "test/splade-model",
        "ENABLE_SPARSE_EMBEDDINGS": "false",
        "RRF_FUSION_WEIGHT_DENSE": "0.8",
        "GPU_ACCELERATION": "false",
    },
)
def test_environment_variable_overrides():
    """Test environment variable overrides for new settings.

    Verifies that all new configuration parameters can be
    properly overridden via environment variables.
    """
    settings = AppSettings()

    assert settings.sparse_embedding_model == "test/splade-model"
    assert settings.enable_sparse_embeddings is False
    assert settings.rrf_fusion_weight_dense == 0.8
    assert settings.gpu_acceleration is False


def test_qdrant_url_configuration():
    """Test Qdrant URL configuration.

    Verifies the Qdrant URL default setting.
    """
    settings = AppSettings()

    assert settings.qdrant_url == "http://localhost:6333"

    # Test environment override
    with patch.dict(os.environ, {"QDRANT_URL": "http://qdrant:6333"}):
        settings = AppSettings()
        assert settings.qdrant_url == "http://qdrant:6333"


def test_settings_field_validation_ranges():
    """Test field validation ranges and constraints.

    Comprehensive validation testing for all numeric fields
    with proper boundary conditions.
    """
    # Test context_size validation
    with pytest.raises(ValidationError):
        AppSettings(context_size=0)  # Should be >= 1

    # Test dense_embedding_dimension validation
    with pytest.raises(ValidationError):
        AppSettings(dense_embedding_dimension=0)  # Should be >= 1

    # Test cuda_device_id validation
    with pytest.raises(ValidationError):
        AppSettings(cuda_device_id=-1)  # Should be >= 0

    # Test prefetch_factor validation
    with pytest.raises(ValidationError):
        AppSettings(prefetch_factor=0)  # Should be >= 1

    with pytest.raises(ValidationError):
        AppSettings(prefetch_factor=10)  # Should be <= 8

    # Test RRF alpha validation
    with pytest.raises(ValidationError):
        AppSettings(rrf_fusion_alpha=0)  # Should be >= 1

    # Test similarity_top_k validation
    with pytest.raises(ValidationError):
        AppSettings(similarity_top_k=0)  # Should be >= 1

    with pytest.raises(ValidationError):
        AppSettings(similarity_top_k=100)  # Should be <= 50


def test_settings_post_init_hook():
    """Test model post-initialization processing.

    Verifies that compatibility aliases are properly set up
    during model initialization.
    """
    settings = AppSettings()

    # Verify alias is set correctly
    assert settings.reranker_model is not None
    assert settings.reranker_model == settings.default_reranker_model

    # Test custom reranker model override
    custom_settings = AppSettings(reranker_model="custom/reranker")
    assert custom_settings.reranker_model == "custom/reranker"


def test_model_config_settings():
    """Test Pydantic model configuration.

    Verifies that the model configuration is properly set
    for environment loading and validation behavior.
    """
    settings = AppSettings()
    config = settings.model_config

    assert config.get("env_file") == ".env"
    assert config.get("env_ignore_empty") is True
    assert config.get("case_sensitive") is False
    assert config.get("env_prefix") == ""
    assert config.get("extra") == "ignore"


def test_analysis_output_field_descriptions():
    """Test AnalysisOutput field descriptions and validation.

    Verifies that all fields have proper descriptions and
    type validation for LLM output parsing.
    """
    # Test valid creation
    output = AnalysisOutput(
        summary="Test summary of document content",
        key_insights=["Insight 1", "Insight 2", "Insight 3"],
        action_items=["Action 1", "Action 2"],
        open_questions=["Question 1", "Question 2"],
    )

    assert len(output.key_insights) == 3
    assert len(output.action_items) == 2
    assert len(output.open_questions) == 2

    # Test empty lists are allowed
    minimal_output = AnalysisOutput(
        summary="Minimal summary", key_insights=[], action_items=[], open_questions=[]
    )

    assert isinstance(minimal_output.key_insights, list)
    assert len(minimal_output.key_insights) == 0

    # Test type validation
    with pytest.raises(ValidationError):
        AnalysisOutput(
            summary="Valid summary",
            key_insights="Should be list",  # Invalid type
            action_items=[],
            open_questions=[],
        )


@pytest.mark.parametrize(
    ("parse_strategy", "expected"),
    [
        ("hi_res", "hi_res"),
        ("fast", "fast"),
        ("auto", "auto"),
    ],
)
def test_parse_strategy_validation(parse_strategy: str, expected: str):
    """Test document parsing strategy configuration.

    Parametrized test for different parsing strategies supported
    by the Unstructured library for document processing.
    """
    settings = AppSettings(parse_strategy=parse_strategy)
    assert settings.parse_strategy == expected


@pytest.mark.parametrize(
    ("quantization_type", "expected"),
    [
        ("int8", "int8"),
        ("int4", "int4"),
        ("float16", "float16"),
    ],
)
def test_quantization_type_validation(quantization_type: str, expected: str):
    """Test quantization type configuration.

    Parametrized test for different quantization types supported
    for memory optimization during embedding computation.
    """
    settings = AppSettings(quantization_type=quantization_type)
    assert settings.quantization_type == expected


# Additional comprehensive tests for enhanced coverage


def test_app_settings_clean_defaults(clean_env):
    """Test AppSettings with clean environment to verify actual defaults."""
    settings = AppSettings()

    # Core configuration defaults
    assert settings.backend == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.default_model == "google/gemma-3n-E4B-it"
    assert settings.context_size == 4096
    assert settings.qdrant_url == "http://localhost:6333"

    # Dense embedding defaults
    assert settings.dense_embedding_model == "BAAI/bge-large-en-v1.5"
    assert settings.dense_embedding_dimension == 1024

    # Sparse embedding defaults (verify corrected model name)
    assert settings.sparse_embedding_model == "prithivida/Splade_PP_en_v1"
    assert settings.enable_sparse_embeddings is True


@pytest.mark.parametrize(
    ("env_var", "field", "value", "expected"),
    [
        ("PARSE_STRATEGY", "parse_strategy", "fast", "fast"),
        ("CHUNK_SIZE", "chunk_size", "512", 512),
        ("CHUNK_OVERLAP", "chunk_overlap", "100", 100),
        ("MAX_ENTITIES", "max_entities", "25", 25),
        ("GPU_ACCELERATION", "gpu_acceleration", "false", False),
        ("DEBUG_MODE", "debug_mode", "true", True),
        ("ENABLE_SPARSE_EMBEDDINGS", "enable_sparse_embeddings", "false", False),
        ("RRF_FUSION_ALPHA", "rrf_fusion_alpha", "45", 45),
        ("EMBEDDING_BATCH_SIZE", "embedding_batch_size", "64", 64),
    ],
)
def test_environment_variable_loading(clean_env, env_var, field, value, expected):
    """Test comprehensive environment variable loading and type conversion."""
    clean_env.setenv(env_var, value)
    settings = AppSettings()
    assert getattr(settings, field) == expected


def test_splade_model_name_validation():
    """Test SPLADE++ model name is correctly specified (typo fix validation)."""
    settings = AppSettings()

    # Verify the corrected SPLADE++ model name
    assert settings.sparse_embedding_model == "prithivida/Splade_PP_en_v1"

    # Test custom SPLADE model override
    custom_settings = AppSettings(sparse_embedding_model="custom/splade-model")
    assert custom_settings.sparse_embedding_model == "custom/splade-model"


def test_analysis_output_json_serialization():
    """Test AnalysisOutput JSON serialization and deserialization."""
    # Create test instance
    original_output = AnalysisOutput(
        summary="Test document summary with detailed analysis",
        key_insights=["Machine learning insights", "Data quality observations"],
        action_items=["Implement ML pipeline", "Review data sources"],
        open_questions=["What about model accuracy?", "How to handle edge cases?"],
    )

    # Test JSON serialization
    json_data = original_output.model_dump()
    assert isinstance(json_data, dict)
    assert json_data["summary"] == original_output.summary
    assert len(json_data["key_insights"]) == 2

    # Test JSON string serialization
    json_string = original_output.model_dump_json()
    assert isinstance(json_string, str)

    # Test deserialization from JSON
    parsed_data = json.loads(json_string)
    reconstructed_output = AnalysisOutput(**parsed_data)

    assert reconstructed_output.summary == original_output.summary
    assert reconstructed_output.key_insights == original_output.key_insights
    assert reconstructed_output.action_items == original_output.action_items
    assert reconstructed_output.open_questions == original_output.open_questions


def test_rrf_fusion_weight_sum_validation():
    """Test RRF fusion weights sum to 1.0 and boundary validation."""
    settings = AppSettings()

    # Default weights should sum to 1.0
    weight_sum = settings.rrf_fusion_weight_dense + settings.rrf_fusion_weight_sparse
    assert abs(weight_sum - 1.0) < 0.001

    # Test custom weights that sum to 1.0
    custom_settings = AppSettings(
        rrf_fusion_weight_dense=0.6, rrf_fusion_weight_sparse=0.4
    )
    custom_sum = (
        custom_settings.rrf_fusion_weight_dense
        + custom_settings.rrf_fusion_weight_sparse
    )
    assert abs(custom_sum - 1.0) < 0.001

    # Test boundary conditions
    boundary_settings = AppSettings(
        rrf_fusion_weight_dense=1.0, rrf_fusion_weight_sparse=0.0
    )
    assert boundary_settings.rrf_fusion_weight_dense == 1.0
    assert boundary_settings.rrf_fusion_weight_sparse == 0.0


def test_all_numeric_field_boundaries():
    """Test boundary conditions for all numeric fields comprehensively."""
    # Test minimum valid values
    min_settings = AppSettings(
        context_size=1,
        dense_embedding_dimension=1,
        cuda_device_id=0,
        embedding_batch_size=1,
        prefetch_factor=1,
        max_concurrent_requests=1,
        reranking_top_k=5,
        chunk_size=256,
        chunk_overlap=0,
        max_entities=1,
        rrf_fusion_alpha=1,
        similarity_top_k=1,
    )
    assert min_settings.context_size == 1
    assert min_settings.embedding_batch_size == 1

    # Test maximum valid values
    max_settings = AppSettings(
        embedding_batch_size=512,
        prefetch_factor=8,
        max_concurrent_requests=100,
        reranking_top_k=100,
        chunk_size=4096,
        chunk_overlap=512,
        max_entities=200,
        similarity_top_k=50,
    )
    assert max_settings.embedding_batch_size == 512
    assert max_settings.prefetch_factor == 8


def test_boundary_validation_failures():
    """Test that invalid boundary values properly raise ValidationError."""
    # Test invalid minimum values
    invalid_configs = [
        {"context_size": 0},
        {"dense_embedding_dimension": 0},
        {"cuda_device_id": -1},
        {"embedding_batch_size": 0},
        {"prefetch_factor": 0},
        {"max_concurrent_requests": 0},
        {"reranking_top_k": 4},
        {"chunk_size": 100},
        {"max_entities": 0},
        {"rrf_fusion_alpha": 0},
        {"similarity_top_k": 0},
    ]

    for config in invalid_configs:
        with pytest.raises(ValidationError):
            AppSettings(**config)

    # Test invalid maximum values
    invalid_max_configs = [
        {"embedding_batch_size": 1000},
        {"prefetch_factor": 10},
        {"max_concurrent_requests": 200},
        {"reranking_top_k": 150},
        {"chunk_size": 8192},
        {"chunk_overlap": 1000},
        {"max_entities": 300},
        {"similarity_top_k": 100},
    ]

    for config in invalid_max_configs:
        with pytest.raises(ValidationError):
            AppSettings(**config)


def test_rrf_fusion_weight_boundaries():
    """Test RRF fusion weight boundary validation."""
    # Test invalid weight ranges (< 0.0 or > 1.0)
    with pytest.raises(ValidationError):
        AppSettings(rrf_fusion_weight_dense=-0.1)

    with pytest.raises(ValidationError):
        AppSettings(rrf_fusion_weight_dense=1.1)

    with pytest.raises(ValidationError):
        AppSettings(rrf_fusion_weight_sparse=-0.1)

    with pytest.raises(ValidationError):
        AppSettings(rrf_fusion_weight_sparse=1.1)


def test_analysis_output_empty_lists():
    """Test AnalysisOutput handles empty lists correctly."""
    empty_output = AnalysisOutput(
        summary="Empty analysis", key_insights=[], action_items=[], open_questions=[]
    )

    assert isinstance(empty_output.key_insights, list)
    assert len(empty_output.key_insights) == 0
    assert isinstance(empty_output.action_items, list)
    assert len(empty_output.action_items) == 0
    assert isinstance(empty_output.open_questions, list)
    assert len(empty_output.open_questions) == 0


def test_analysis_output_field_validation_errors():
    """Test AnalysisOutput field validation with invalid types."""
    with pytest.raises(ValidationError) as exc_info:
        AnalysisOutput(
            summary=123,  # Should be string
            key_insights=[],
            action_items=[],
            open_questions=[],
        )
    assert "summary" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        AnalysisOutput(
            summary="Valid summary",
            key_insights="Not a list",  # Should be list
            action_items=[],
            open_questions=[],
        )
    assert "key_insights" in str(exc_info.value)
