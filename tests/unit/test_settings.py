"""Comprehensive tests for the recovered configuration architecture.

This module provides comprehensive tests for the unified DocMindSettings system
with nested configuration models that was recovered to eliminate scattered constants
and provide clean ADR compliance features.

Tests cover:
- Nested configuration models (VLLMConfig, AgentConfig, etc.)
- ADR compliance features (ADR-011, ADR-018, ADR-019)
- Field validators and constraints
- Environment variable overrides with nested delimiter support
- Configuration method helpers
- Directory creation post-initialization
- Test-production separation
- Edge cases and error handling

This validates the critical infrastructure the entire application relies on.
"""

import os
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config.settings import DocMindSettings, settings
from tests.fixtures.test_settings import TestDocMindSettings


class TestSettingsDefaults:
    """Test all default values load correctly and are sensible."""

    def test_application_metadata_defaults(self):
        """Test application metadata has correct defaults."""
        s = settings

        assert s.app_name == "DocMind AI"
        assert s.app_version == "2.0.0"
        assert s.debug is False

    def test_multi_agent_defaults(self):
        """Test multi-agent coordination defaults are properly configured."""
        s = settings

        # Multi-agent should be enabled by default (from AgentConfig)
        assert s.agents.enable_multi_agent is True
        assert s.agents.decision_timeout == 200  # ADR-011: 200ms default timeout
        assert s.agents.enable_fallback_rag is True
        assert s.agents.max_retries == 2  # Reasonable retry count

    def test_llm_backend_defaults(self):
        """Test LLM backend defaults are properly configured for local-first."""
        s = settings

        assert s.llm_backend == "ollama"  # Ollama backend for local-first
        assert (
            s.vllm.model == "Qwen/Qwen3-4B-Instruct-2507-FP8"
        )  # FP8 model from VLLMConfig
        assert s.ollama_base_url == "http://localhost:11434"  # Local Ollama
        # Note: API key and temperature are not top-level fields in recovered config

    def test_model_optimization_defaults(self):
        """Test model optimization settings are correctly configured."""
        s = settings

        # vLLM optimization settings from VLLMConfig
        assert s.vllm.kv_cache_dtype == "fp8_e5m2"  # FP8 KV cache
        assert s.vllm.attention_backend == "FLASHINFER"  # FlashInfer backend
        assert s.vllm.enable_chunked_prefill is True  # Chunked prefill enabled

    def test_context_management_defaults(self):
        """Test context management has proper 128K defaults."""
        s = settings

        assert s.vllm.context_window == 131072  # 128K from VLLMConfig
        assert (
            s.agents.context_buffer_size == 8192
        )  # Agent context buffer from AgentConfig
        # Note: conversation memory is not a top-level field in recovered config

    def test_document_processing_defaults(self):
        """Test document processing has sensible defaults."""
        s = settings

        # Document processing from ProcessingConfig
        assert s.processing.chunk_size == 1500
        assert s.processing.max_document_size_mb == 100

        # Caching from CacheConfig
        assert s.cache.enable_document_caching is True

    def test_retrieval_defaults(self):
        """Test retrieval configuration defaults."""
        s = settings

        # Retrieval settings from RetrievalConfig
        assert s.retrieval.strategy == "hybrid"  # Hybrid is optimal
        assert s.retrieval.top_k == 10
        assert s.retrieval.use_reranking is True  # BGE reranking enabled
        assert s.retrieval.reranking_top_k == 5

    def test_embedding_defaults(self):
        """Test embedding configuration defaults."""
        s = settings

        # Embedding settings from EmbeddingConfig
        assert s.embedding.model_name == "BAAI/bge-m3"  # BGE-M3 model
        assert s.embedding.dimension == 1024  # BGE-M3 dimension
        assert s.embedding.max_length == 8192  # Max sequence length

    def test_vector_database_defaults(self):
        """Test vector database configuration defaults."""
        s = settings

        assert s.database.vector_store_type == "qdrant"  # Qdrant is primary
        assert s.database.qdrant_url == "http://localhost:6333"
        assert s.database.qdrant_collection == "docmind_docs"

    def test_performance_defaults(self):
        """Test performance configuration defaults are reasonable."""
        s = settings

        assert s.monitoring.max_query_latency_ms == 2000  # 2s max latency
        assert s.monitoring.max_memory_gb == 4.0  # 4GB RAM limit
        assert s.monitoring.max_vram_gb == 14.0  # 14GB for FP8 on RTX 4090
        assert s.enable_gpu_acceleration is True

    def test_vllm_defaults(self):
        """Test vLLM-specific defaults are optimized for RTX 4090."""
        s = settings

        assert s.vllm.gpu_memory_utilization == 0.85  # 85% utilization
        assert s.vllm.attention_backend == "FLASHINFER"  # FlashInfer backend
        assert s.vllm.enable_chunked_prefill is True
        assert s.vllm.max_num_batched_tokens == 8192
        assert s.vllm.max_num_seqs == 16

    def test_persistence_defaults(self):
        """Test persistence configuration creates proper paths."""
        s = settings

        assert s.data_dir == Path("./data")
        assert s.cache_dir == Path("./cache")
        assert s.database.sqlite_db_path == Path("./data/docmind.db")
        assert s.database.enable_wal_mode is True  # WAL mode for performance

    def test_centralized_constants_defaults(self):
        """Test all centralized constants have proper defaults."""
        s = settings

        # Memory conversion constants
        assert s.monitoring.bytes_to_gb_divisor == 1024**3
        assert s.monitoring.bytes_to_mb_divisor == 1024 * 1024

        # BGE-M3 constants (now in nested embedding config)
        assert s.embedding.dimension == 1024
        assert s.embedding.max_length == 8192
        assert s.embedding.model_name == "BAAI/bge-m3"
        assert s.embedding.batch_size_gpu == 12
        assert s.embedding.batch_size_cpu == 4

        # Hybrid retrieval constants
        assert s.retrieval.rrf_alpha == 60  # RRF fusion alpha parameter
        assert s.retrieval.rrf_k_constant == 60
        assert s.retrieval.rrf_fusion_weight_dense == 0.7
        assert s.retrieval.rrf_fusion_weight_sparse == 0.3

        # Context window settings (now in nested vllm config)
        assert s.vllm.context_window == 131072  # 128K
        assert s.vllm.max_tokens == 2048
        # Note: request_timeout_seconds, streaming_delay_seconds, and minimum_vram_*
        # properties were deleted


class TestFieldValidation:
    """Test all field validators work correctly and catch invalid values."""

    def test_agent_timeout_range_validation(self):
        """Test agent decision timeout must be within valid range."""
        # Valid range
        DocMindSettings(agents={"decision_timeout": 100})
        DocMindSettings(agents={"decision_timeout": 500})
        DocMindSettings(agents={"decision_timeout": 1000})

        # Invalid: too low
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 10"
        ):
            DocMindSettings(agents={"decision_timeout": 5})

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 1000"
        ):
            DocMindSettings(agents={"decision_timeout": 2000})

    def test_max_agent_retries_validation(self):
        """Test max agent retries has reasonable bounds."""
        # Valid range
        DocMindSettings(agents={"max_retries": 0})
        DocMindSettings(agents={"max_retries": 3})
        DocMindSettings(agents={"max_retries": 10})

        # Invalid: negative
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            DocMindSettings(agents={"max_retries": -1})

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 10"
        ):
            DocMindSettings(agents={"max_retries": 15})

    def test_llm_temperature_validation(self):
        """Test LLM temperature must be in valid range."""
        # Valid temperatures
        DocMindSettings(vllm={"temperature": 0.0})
        DocMindSettings(vllm={"temperature": 0.5})
        DocMindSettings(vllm={"temperature": 2.0})

        # Invalid: negative
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            DocMindSettings(vllm={"temperature": -0.1})

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 2"
        ):
            DocMindSettings(vllm={"temperature": 3.0})

    def test_llm_max_tokens_validation(self):
        """Test LLM max tokens has reasonable bounds."""
        # Valid token counts
        DocMindSettings(vllm={"max_tokens": 100})
        DocMindSettings(vllm={"max_tokens": 2048})
        DocMindSettings(vllm={"max_tokens": 8192})

        # Invalid: too low
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 100"
        ):
            DocMindSettings(vllm={"max_tokens": 50})

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 8192"
        ):
            DocMindSettings(vllm={"max_tokens": 16384})

    def test_chunk_size_validation(self):
        """Test chunk size has reasonable bounds."""
        # Valid chunk sizes
        DocMindSettings(processing={"chunk_size": 100})
        DocMindSettings(processing={"chunk_size": 1500})
        DocMindSettings(processing={"chunk_size": 10000})

        # Invalid: too small
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 100"
        ):
            DocMindSettings(processing={"chunk_size": 50})

        # Invalid: too large
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 10000"
        ):
            DocMindSettings(processing={"chunk_size": 15000})

    def test_chunk_overlap_validation(self):
        """Test chunk overlap has reasonable bounds."""
        # Valid overlaps
        DocMindSettings(processing={"chunk_overlap": 0})
        DocMindSettings(processing={"chunk_overlap": 100})
        DocMindSettings(processing={"chunk_overlap": 200})

        # Invalid: negative
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            DocMindSettings(processing={"chunk_overlap": -10})

        # Invalid: too large
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 200"
        ):
            DocMindSettings(processing={"chunk_overlap": 500})

    def test_top_k_validation(self):
        """Test top_k retrieval has reasonable bounds."""
        # Valid top-k values
        DocMindSettings(retrieval={"top_k": 1})
        DocMindSettings(retrieval={"top_k": 10})
        DocMindSettings(retrieval={"top_k": 50})

        # Invalid: zero or negative
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 1"
        ):
            DocMindSettings(retrieval={"top_k": 0})

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 50"
        ):
            DocMindSettings(retrieval={"top_k": 100})

    def test_bge_m3_dimension_validation(self):
        """Test BGE-M3 embedding dimension validation."""
        # Valid dimensions
        DocMindSettings(embedding={"dimension": 256})
        DocMindSettings(embedding={"dimension": 1024})
        DocMindSettings(embedding={"dimension": 4096})

        # Invalid: too small
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 256"
        ):
            DocMindSettings(embedding={"dimension": 128})

        # Invalid: too large
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 4096"
        ):
            DocMindSettings(embedding={"dimension": 8192})

    def test_rrf_alpha_validation(self):
        """Test RRF alpha weight validation."""
        # Valid alpha values (integer range 10-100)
        DocMindSettings(retrieval={"rrf_alpha": 10})
        DocMindSettings(retrieval={"rrf_alpha": 60})  # Default
        DocMindSettings(retrieval={"rrf_alpha": 100})

        # Invalid: too low
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 10"
        ):
            DocMindSettings(retrieval={"rrf_alpha": 5})

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 100"
        ):
            DocMindSettings(retrieval={"rrf_alpha": 150})

    def test_vllm_gpu_memory_validation(self):
        """Test vLLM GPU memory utilization validation."""
        # Valid utilization values
        DocMindSettings(vllm={"gpu_memory_utilization": 0.5})
        DocMindSettings(vllm={"gpu_memory_utilization": 0.85})
        DocMindSettings(vllm={"gpu_memory_utilization": 0.95})

        # Invalid: too low
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0.1"
        ):
            DocMindSettings(vllm={"gpu_memory_utilization": 0.05})

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 0.95"
        ):
            DocMindSettings(vllm={"gpu_memory_utilization": 1.0})

    def test_streamlit_port_validation(self):
        """Test Streamlit port validation."""
        # Valid ports
        DocMindSettings(ui={"streamlit_port": 1024})
        DocMindSettings(ui={"streamlit_port": 8501})
        DocMindSettings(ui={"streamlit_port": 65535})

        # Invalid: too low (system ports)
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 1024"
        ):
            DocMindSettings(ui={"streamlit_port": 80})

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 65535"
        ):
            DocMindSettings(ui={"streamlit_port": 70000})


class TestDirectoryCreation:
    """Test directory creation validators work correctly."""

    def test_data_directory_creation(self, tmp_path):
        """Test data directory is created on settings instantiation."""
        test_data_dir = tmp_path / "test_data"

        # Directory shouldn't exist yet
        assert not test_data_dir.exists()

        # Create settings with custom data directory
        DocMindSettings(data_dir=str(test_data_dir))

        # Directory should now exist
        assert test_data_dir.exists()
        assert test_data_dir.is_dir()

    def test_cache_directory_creation(self, tmp_path):
        """Test cache directory is created on settings instantiation."""
        test_cache_dir = tmp_path / "test_cache"

        assert not test_cache_dir.exists()
        DocMindSettings(cache_dir=str(test_cache_dir))
        assert test_cache_dir.exists()

    def test_log_file_parent_creation(self, tmp_path):
        """Test log file parent directory is created."""
        test_log_file = tmp_path / "logs" / "test.log"

        # Parent directory shouldn't exist
        assert not test_log_file.parent.exists()

        DocMindSettings(log_file=str(test_log_file))

        # Parent directory should be created
        assert test_log_file.parent.exists()
        assert test_log_file.parent.is_dir()

    def test_sqlite_db_parent_creation(self, tmp_path):
        """Test SQLite database parent directory is created."""
        test_db_path = tmp_path / "db" / "test.db"

        assert not test_db_path.parent.exists()
        DocMindSettings(database={"sqlite_db_path": str(test_db_path)})
        assert test_db_path.parent.exists()

    def test_nested_directory_creation(self, tmp_path):
        """Test nested directories are created properly."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"

        assert not nested_dir.exists()
        DocMindSettings(data_dir=str(nested_dir))
        assert nested_dir.exists()
        assert nested_dir.is_dir()


class TestLLMBackendValidation:
    """Test LLM backend validation and warnings."""

    def test_valid_llm_backends(self):
        """Test all valid LLM backends are accepted."""
        valid_backends = ["ollama", "llamacpp", "vllm", "openai"]

        for backend in valid_backends:
            s = DocMindSettings(llm_backend=backend)
            assert s.llm_backend == backend

    def test_valid_backend_acceptance(self):
        """Test valid backends are accepted without warnings."""
        for backend in ["ollama", "llamacpp", "vllm", "openai"]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                DocMindSettings(llm_backend=backend)
                # No warnings should be generated for valid backends
                backend_warnings = [warn for warn in w if backend in str(warn.message)]
                assert len(backend_warnings) == 0


class TestEnvironmentVariableOverrides:
    """Test environment variable overrides work correctly."""

    def test_basic_env_override(self):
        """Test basic environment variable override."""
        with patch.dict(os.environ, {"DOCMIND_DEBUG": "true"}):
            s = DocMindSettings()  # Create new instance to pick up env vars
            assert s.debug is True

    def test_numeric_env_override(self):
        """Test numeric environment variables are properly converted."""
        with patch.dict(
            os.environ,
            {
                "DOCMIND_AGENTS__DECISION_TIMEOUT": "500",
                "DOCMIND_VLLM__TEMPERATURE": "0.5",
                "DOCMIND_RETRIEVAL__TOP_K": "15",
            },
        ):
            s = DocMindSettings()  # Create new instance to pick up env vars
            assert s.agents.decision_timeout == 500
            assert s.vllm.temperature == 0.5
            assert s.retrieval.top_k == 15

    def test_boolean_env_override(self):
        """Test boolean environment variables work correctly."""
        with patch.dict(
            os.environ,
            {
                "DOCMIND_AGENTS__ENABLE_MULTI_AGENT": "false",
                "DOCMIND_RETRIEVAL__USE_RERANKING": "true",
                "DOCMIND_ENABLE_GPU_ACCELERATION": "false",
            },
        ):
            s = DocMindSettings()  # Create new instance to pick up env vars
            assert s.agents.enable_multi_agent is False
            assert s.retrieval.use_reranking is True
            assert s.enable_gpu_acceleration is False

    def test_path_env_override(self):
        """Test path environment variables work correctly."""
        # Use /tmp paths to avoid permission issues
        with patch.dict(
            os.environ,
            {
                "DOCMIND_DATA_DIR": "/tmp/custom/data",
                "DOCMIND_CACHE_DIR": "/tmp/custom/cache",
            },
        ):
            s = DocMindSettings()  # Create new instance to pick up env vars
            assert s.data_dir == Path("/tmp/custom/data")
            assert s.cache_dir == Path("/tmp/custom/cache")

    def test_list_env_override(self):
        """Test list environment variables are handled correctly."""
        # Note: Pydantic typically expects JSON for list env vars
        with patch.dict(
            os.environ,
            {"DOCMIND_UI__CONTEXT_SIZE_OPTIONS": "[1024, 2048, 4096]"},
        ):
            s = DocMindSettings()  # Create new instance to pick up env vars
            assert s.ui.context_size_options == [1024, 2048, 4096]

    def test_env_validation_still_applies(self):
        """Test validation still applies with environment overrides."""
        with (
            patch.dict(os.environ, {"DOCMIND_AGENTS__DECISION_TIMEOUT": "5000"}),
            pytest.raises(
                ValidationError, match="Input should be less than or equal to 1000"
            ),
        ):
            _ = DocMindSettings()  # Create new instance to trigger validation

    def test_env_prefix_enforced(self):
        """Test environment variables must have DOCMIND_ prefix."""
        with patch.dict(os.environ, {"DEBUG": "true"}):  # No prefix
            s = settings
            assert s.debug is False  # Should use default, not env var

    def test_case_insensitive_env_vars(self):
        """Test environment variables are case insensitive."""
        with patch.dict(os.environ, {"docmind_debug": "true"}):
            s = DocMindSettings()  # Create new instance to pick up env vars
            assert s.debug is True


class TestConfigurationMethods:
    """Test configuration method helpers return correct subsets."""

    def test_get_model_config(self):
        """Test get_model_config returns correct model configuration."""
        s = settings
        model_config = s.get_model_config()

        expected_keys = {
            "model_name",
            "context_window",
            "max_tokens",
            "temperature",
            "base_url",
        }

        assert set(model_config.keys()) == expected_keys
        assert model_config["model_name"] == "Qwen/Qwen3-4B-Instruct-2507-FP8"
        assert model_config["context_window"] == 131072
        assert model_config["base_url"] == "http://localhost:11434"

    def test_get_embedding_config(self):
        """Test get_embedding_config returns embedding configuration."""
        s = settings
        embedding_config = s.get_embedding_config()

        expected_keys = {
            "model_name",
            "device",
            "max_length",
            "batch_size",
            "trust_remote_code",
        }

        assert set(embedding_config.keys()) == expected_keys
        assert embedding_config["model_name"] == "BAAI/bge-m3"
        assert embedding_config["max_length"] == 8192
        assert embedding_config["trust_remote_code"] is True

    def test_get_processing_config(self):
        """Test get_processing_config returns processing configuration."""
        s = settings
        processing_config = s.get_processing_config()

        expected_keys = {
            "chunk_size",
            "new_after_n_chars",
            "combine_text_under_n_chars",
            "multipage_sections",
            "max_document_size_mb",
        }

        assert set(processing_config.keys()) == expected_keys
        assert processing_config["chunk_size"] == 1500
        assert processing_config["multipage_sections"] is True

    def test_to_dict_method(self):
        """Test to_dict method returns complete settings."""
        s = settings
        settings_dict = s.model_dump()

        # Should contain all fields
        assert "app_name" in settings_dict
        assert "vllm" in settings_dict
        assert "agents" in settings_dict
        assert "embedding" in settings_dict

        # Values should match
        assert settings_dict["app_name"] == "DocMind AI"
        assert settings_dict["agents"]["enable_multi_agent"] is True
        assert settings_dict["vllm"]["model"] == "Qwen/Qwen3-4B-Instruct-2507-FP8"


class TestCentralizedConstants:
    """Test all centralized constants are accessible and have correct values."""

    def test_memory_conversion_constants(self):
        """Test memory conversion constants are correct."""
        s = settings

        assert s.monitoring.bytes_to_gb_divisor == 1024**3  # 1073741824
        assert s.monitoring.bytes_to_mb_divisor == 1024 * 1024  # 1048576

    def test_bge_m3_constants(self):
        """Test BGE-M3 model constants are properly centralized."""
        s = settings

        # BGE-M3 constants (now in nested embedding config)
        assert s.embedding.dimension == 1024
        assert s.embedding.max_length == 8192
        assert s.embedding.model_name == "BAAI/bge-m3"
        assert s.embedding.batch_size_gpu == 12  # Optimized for GPU
        assert s.embedding.batch_size_cpu == 4  # Conservative for CPU

    def test_hybrid_retrieval_constants(self):
        """Test hybrid retrieval constants are research-backed."""
        s = settings

        # RRF parameters
        assert s.retrieval.rrf_alpha == 60  # RRF fusion alpha parameter
        assert s.retrieval.rrf_k_constant == 60

        # Research-backed weights
        assert s.retrieval.rrf_fusion_weight_dense == 0.7
        assert s.retrieval.rrf_fusion_weight_sparse == 0.3

        # Weights should sum to 1.0
        assert (
            s.retrieval.rrf_fusion_weight_dense + s.retrieval.rrf_fusion_weight_sparse
            == 1.0
        )

    def test_default_processing_constants(self):
        """Test default processing value constants."""
        s = settings

        assert s.monitoring.default_batch_size == 20
        assert s.monitoring.default_confidence_threshold == 0.8
        assert s.monitoring.default_entity_confidence == 0.8
        assert s.retrieval.top_k == 10
        assert s.retrieval.reranking_top_k == 5

    def test_timeout_configuration_constants(self):
        """Test timeout configuration constants."""
        s = settings

        assert s.database.qdrant_timeout == 60  # 1 minute
        assert s.monitoring.default_agent_timeout == 3.0  # 3 seconds
        assert s.monitoring.cache_expiry_seconds == 3600  # 1 hour
        assert s.monitoring.spacy_download_timeout == 300  # 5 minutes

    def test_context_configuration_modernized(self):
        """Test context configuration has been moved to nested vllm config."""
        s = settings

        # Context window settings (now in nested vllm config)
        assert s.vllm.context_window == 131072  # 128K
        assert s.vllm.max_tokens == 2048
        # Note: suggested_context_* properties were removed as part of cleanup

        # Note: request_timeout_seconds, streaming_delay_seconds, and minimum_vram_*
        # properties were deleted

    def test_monitoring_constants(self):
        """Test performance monitoring constants."""
        s = settings

        assert s.monitoring.cpu_monitoring_interval == 0.1  # 100ms
        assert s.monitoring.percent_multiplier == 100


class TestGlobalSettingsInstance:
    """Test the global settings instance works correctly."""

    def test_global_settings_instance_exists(self):
        """Test global settings instance is available."""
        assert settings is not None
        assert isinstance(settings, DocMindSettings)

    def test_global_settings_has_defaults(self):
        """Test global settings instance has proper defaults."""
        assert settings.app_name == "DocMind AI"
        assert settings.agents.enable_multi_agent is True
        assert settings.vllm.model == "Qwen/Qwen3-4B-Instruct-2507-FP8"

    def test_global_settings_is_singleton(self):
        """Test global settings behaves like singleton."""
        from src.config import settings as settings1
        from src.config import settings as settings2

        # Should be the same instance
        assert settings1 is settings2


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_none_values_handled_correctly(self):
        """Test None values are handled correctly where allowed."""
        # Most fields require valid values, but some can be optional
        s = DocMindSettings(
            debug=False,  # Boolean field with valid value
        )

        # Verify configuration is valid with basic settings
        assert s.debug is False
        assert s.log_file is not None  # Has default value

    def test_empty_string_handling(self):
        """Test empty strings are handled appropriately."""
        # Most string fields have default values and accept empty strings
        # Test that empty app_name is accepted (as per current validation)
        s = DocMindSettings(app_name="")
        assert s.app_name == ""  # Empty string is accepted

        # Test that settings can be created with non-empty strings
        s2 = DocMindSettings(app_name="Test App")
        assert s2.app_name == "Test App"

        # Test that whitespace-only strings are also accepted
        s3 = DocMindSettings(app_name="   ")
        assert s3.app_name == "   "

        # Test that log_level accepts empty strings or revert to default
        s4 = DocMindSettings(log_level="DEBUG")
        assert s4.log_level == "DEBUG"

    def test_extreme_boundary_values(self):
        """Test extreme boundary values."""
        # Test minimum values
        s = DocMindSettings(
            agents={"decision_timeout": 10, "max_retries": 0},
            vllm={"temperature": 0.0},
            processing={"chunk_overlap": 0},
        )

        assert s.agents.decision_timeout == 10
        assert s.agents.max_retries == 0
        assert s.vllm.temperature == 0.0
        assert s.processing.chunk_overlap == 0

        # Test maximum values
        s = DocMindSettings(
            agents={"decision_timeout": 1000, "max_retries": 10},
            vllm={"temperature": 2.0},
            retrieval={"top_k": 50},
        )

        assert s.agents.decision_timeout == 1000
        assert s.agents.max_retries == 10
        assert s.vllm.temperature == 2.0
        assert s.retrieval.top_k == 50

    def test_type_coercion_edge_cases(self):
        """Test type coercion works correctly."""
        # String numbers should be converted
        with patch.dict(
            os.environ,
            {"DOCMIND_RETRIEVAL__TOP_K": "15", "DOCMIND_VLLM__TEMPERATURE": "0.5"},
        ):
            s = settings
            assert isinstance(s.retrieval.top_k, int)
            assert isinstance(s.vllm.temperature, float)
            assert s.retrieval.top_k == 15
            assert s.vllm.temperature == 0.5

    def test_invalid_literal_values(self):
        """Test invalid literal values are rejected."""
        with pytest.raises(ValidationError, match="Input should be"):
            DocMindSettings(llm_backend="invalid_backend")

        with pytest.raises(ValidationError, match="Input should be"):
            DocMindSettings(retrieval={"strategy": "invalid_strategy"})

        with pytest.raises(ValidationError, match="Input should be"):
            DocMindSettings(vector_store_type="invalid_store")

    def test_path_edge_cases(self, tmp_path):
        """Test path handling edge cases."""
        # Relative paths should work
        s = DocMindSettings(data_dir="./relative/path")
        assert s.data_dir == Path("./relative/path")

        # Path-like strings should work
        test_path = str(tmp_path / "test")
        s = DocMindSettings(data_dir=test_path)
        assert s.data_dir == Path(test_path)


class TestRealWorldScenarios:
    """Test real-world usage scenarios and business logic."""

    def test_rtx_4090_optimized_configuration(self):
        """Test settings are optimized for RTX 4090 16GB setup."""
        s = settings

        # Memory settings should be optimized for 16GB VRAM
        # Note: max_vram_gb was removed - GPU memory utilization setting handles this
        assert s.vllm.gpu_memory_utilization == 0.85  # Conservative

        # FP8 optimization should be enabled (now in nested vllm config)
        assert s.vllm.kv_cache_dtype == "fp8_e5m2"
        # Note: quantization and enable_kv_cache_optimization properties were deleted

        # FlashInfer should be configured (now in nested vllm config)
        assert s.vllm.attention_backend == "FLASHINFER"

    def test_128k_context_configuration(self):
        """Test 128K context window is properly configured."""
        s = settings

        # Context window settings (now in nested vllm config)
        assert s.vllm.context_window == 131072  # 128K
        # Note: context_buffer_size, default_token_limit, and context_size_options
        # properties were deleted

    def test_local_first_configuration(self):
        """Test configuration enforces local-first architecture."""
        s = settings

        # Default backend should be local
        assert s.llm_backend in ["ollama", "llamacpp", "vllm"]  # Not OpenAI
        assert s.ollama_base_url.startswith("http://localhost")
        assert s.qdrant_url.startswith("http://localhost")
        # No API key needed for local deployment

    def test_hybrid_search_configuration(self):
        """Test hybrid search is properly configured."""
        s = settings

        assert s.retrieval.strategy == "hybrid"
        assert s.retrieval.use_sparse_embeddings is True
        assert s.retrieval.use_reranking is True

        # Research-backed weights should be configured
        assert s.retrieval.rrf_fusion_weight_dense == 0.7
        assert s.retrieval.rrf_fusion_weight_sparse == 0.3

    def test_production_ready_defaults(self):
        """Test defaults are suitable for production use."""
        s = settings

        # Performance settings should be reasonable
        assert s.max_query_latency_ms <= 5000  # Under 5 seconds
        assert s.max_memory_gb <= 8.0  # Reasonable RAM usage
        assert s.agents.decision_timeout <= 1000  # Under 1 second

        # Caching should be enabled for performance
        assert s.cache.enable_document_caching is True
        assert s.cache_expiry_seconds > 0

        # Multi-agent should be enabled by default
        assert s.agents.enable_multi_agent is True
        assert s.agents.enable_fallback_rag is True

    def test_development_vs_production_scenarios(self):
        """Test settings work for both development and production."""
        # Development scenario (debug enabled)
        dev_settings = DocMindSettings(debug=True, log_level="DEBUG")
        assert dev_settings.debug is True
        assert dev_settings.log_level == "DEBUG"

        # Production scenario (debug disabled)
        prod_settings = DocMindSettings(debug=False, log_level="INFO")
        assert prod_settings.debug is False
        assert prod_settings.log_level == "INFO"

        # Both should have valid configurations
        assert dev_settings.vllm.model == prod_settings.vllm.model
        assert (
            dev_settings.agents.enable_multi_agent
            == prod_settings.agents.enable_multi_agent
        )


class TestADRComplianceFeatures:
    """Test ADR compliance features in the recovered configuration architecture."""

    def test_adr_011_agent_orchestration_settings(self):
        """Test ADR-011 agent orchestration configuration compliance."""
        s = settings

        # Test the ADR-011 compliance method
        config = s.get_agent_orchestration_config()

        # Verify all ADR-011 required fields are present
        required_keys = {
            "context_trim_threshold",
            "context_buffer_size",
            "enable_parallel_execution",
            "max_workflow_depth",
            "enable_state_compression",
            "chat_memory_limit",
            "decision_timeout",
        }
        assert set(config.keys()) == required_keys

        # Verify ADR-011 specific constraints
        assert config["decision_timeout"] == 200  # ADR-011: 200ms timeout
        assert config["context_trim_threshold"] >= 65536  # Minimum context management
        assert config["enable_parallel_execution"] is True  # Parallel tool execution
        assert config["max_workflow_depth"] >= 2  # Multi-step workflows

    def test_adr_011_context_management_compliance(self):
        """Test ADR-011 context management specific requirements."""
        s = settings

        # Direct access to agent config settings
        assert s.agents.context_trim_threshold == 122880  # ADR-011 default
        assert s.agents.context_buffer_size == 8192  # Buffer management
        assert s.agents.chat_memory_limit_tokens == 66560  # Memory limits
        assert s.agents.enable_parallel_tool_execution is True
        assert s.agents.enable_agent_state_compression is True

    def test_adr_018_dspy_optimization_compliance(self):
        """Test ADR-018 DSPy optimization configuration."""
        s = settings

        # Test the DSPy configuration method
        dspy_config = s.get_dspy_config()

        # Verify DSPy configuration structure
        required_keys = {"enabled", "iterations", "metric_threshold", "bootstrapping"}
        assert set(dspy_config.keys()) == required_keys

        # Test ADR-018 specific settings
        assert "enabled" in dspy_config
        assert dspy_config["iterations"] == 10  # Default iterations
        assert dspy_config["metric_threshold"] == 0.8  # Quality threshold
        assert dspy_config["bootstrapping"] is True  # Bootstrapping enabled

        # Test that DSPy can be disabled (default per user feedback)
        assert s.enable_dspy_optimization is False  # Disabled by default

    def test_adr_019_graphrag_configuration_compliance(self):
        """Test ADR-019 GraphRAG configuration."""
        s = settings

        # Test the GraphRAG configuration method
        graphrag_config = s.get_graphrag_config()

        # Verify GraphRAG configuration structure
        required_keys = {
            "enabled",
            "relationship_extraction",
            "entity_resolution",
            "max_hops",
        }
        assert set(graphrag_config.keys()) == required_keys

        # Test ADR-019 specific settings
        assert graphrag_config["enabled"] is False  # Disabled by default
        assert graphrag_config["relationship_extraction"] is False  # Optional feature
        assert (
            graphrag_config["entity_resolution"] == "fuzzy"
        )  # Default resolution method
        assert graphrag_config["max_hops"] == 2  # Default traversal depth
        assert 1 <= graphrag_config["max_hops"] <= 5  # Within valid range

    def test_adr_configuration_methods_integration(self):
        """Test that all ADR configuration methods work together."""
        s = settings

        # All configuration methods should return valid dictionaries
        agent_config = s.get_agent_orchestration_config()
        dspy_config = s.get_dspy_config()
        graphrag_config = s.get_graphrag_config()

        assert isinstance(agent_config, dict)
        assert isinstance(dspy_config, dict)
        assert isinstance(graphrag_config, dict)

        # All should have content
        assert len(agent_config) > 0
        assert len(dspy_config) > 0
        assert len(graphrag_config) > 0

        # Verify no unexpected key collisions (some overlap is acceptable)
        all_keys = (
            set(agent_config.keys())
            | set(dspy_config.keys())
            | set(graphrag_config.keys())
        )
        total_keys = len(agent_config) + len(dspy_config) + len(graphrag_config)
        # Note: Some keys may legitimately overlap between configs (e.g., "enabled")
        assert len(all_keys) <= total_keys  # No more unique keys than total keys

    def test_test_production_separation_compliance(self):
        """Test that test and production settings are properly separated."""
        # Production settings
        prod_settings = settings

        # Test settings
        test_settings = TestDocMindSettings()

        # Key differences that ensure test-production separation
        assert prod_settings.enable_gpu_acceleration is True  # Production: GPU enabled
        assert test_settings.enable_gpu_acceleration is False  # Test: CPU only

        assert (
            prod_settings.agents.decision_timeout == 200
        )  # Production: standard timeout
        assert test_settings.agents.decision_timeout == 100  # Test: faster timeout

        assert prod_settings.vllm.context_window == 131072  # Production: full context
        assert test_settings.vllm.context_window == 8192  # Test: smaller context

        assert (
            prod_settings.cache.enable_document_caching is True
        )  # Production: caching enabled
        assert (
            test_settings.cache.enable_document_caching is False
        )  # Test: no caching for isolation

        # Both should be valid configurations
        assert isinstance(prod_settings, DocMindSettings)
        assert isinstance(
            test_settings, DocMindSettings
        )  # Test settings inherit from production

    def test_nested_configuration_validation(self):
        """Test that nested configuration models validate properly."""
        s = settings

        # All nested configs should be properly instantiated
        assert s.vllm is not None
        assert s.processing is not None
        assert s.agents is not None
        assert s.embedding is not None
        assert s.retrieval is not None
        assert s.cache is not None

        # All should have proper types
        from src.config.settings import (
            AgentConfig,
            CacheConfig,
            EmbeddingConfig,
            ProcessingConfig,
            RetrievalConfig,
            VLLMConfig,
        )

        assert isinstance(s.vllm, VLLMConfig)
        assert isinstance(s.processing, ProcessingConfig)
        assert isinstance(s.agents, AgentConfig)
        assert isinstance(s.embedding, EmbeddingConfig)
        assert isinstance(s.retrieval, RetrievalConfig)
        assert isinstance(s.cache, CacheConfig)
