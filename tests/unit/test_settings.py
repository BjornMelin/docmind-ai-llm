"""Comprehensive tests for the centralized settings system.

This module provides brutally honest, real-world valuable tests for the
centralized configuration system at src/config/settings.py that was recently
implemented to eliminate DRY violations and centralize scattered constants.

Tests cover:
- All 100+ centralized fields and constants
- Field validators and their business logic
- Type validation and range constraints
- Environment variable overrides
- Configuration methods (get_agent_config, get_performance_config, etc.)
- Directory creation validators
- LLM backend validation warnings
- Edge cases and error handling
- Integration points and backward compatibility

This is critical infrastructure that the entire application relies on.
"""

import os
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config.app_settings import DocMindSettings
from src.config.settings import settings


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

        # Multi-agent should be enabled by default
        assert s.enable_multi_agent is True
        assert s.agent_decision_timeout == 300  # 300ms for fast response
        assert s.enable_fallback_rag is True
        assert s.max_agent_retries == 2  # Reasonable retry count

    def test_llm_backend_defaults(self):
        """Test LLM backend defaults are properly configured for local-first."""
        s = settings

        assert s.llm_backend == "vllm"  # vLLM for FP8 optimization
        assert s.model_name == "Qwen/Qwen3-4B-Instruct-2507"  # Latest model
        assert s.llm_base_url == "http://localhost:11434"  # Local Ollama
        assert s.llm_api_key is None  # No API key for local
        assert s.llm_temperature == 0.1  # Low for consistency
        assert s.llm_max_tokens == 2048  # Reasonable generation length

    def test_model_optimization_defaults(self):
        """Test model optimization settings are correctly configured."""
        s = settings

        assert s.quantization == "fp8"  # FP8 for memory efficiency
        assert s.kv_cache_dtype == "fp8"  # FP8 for KV cache
        assert s.enable_kv_cache_optimization is True
        assert s.kv_cache_performance_boost == 1.3  # 30% boost

    def test_context_management_defaults(self):
        """Test context management has proper 128K defaults."""
        s = settings

        assert s.context_window_size == 131072  # 128K
        assert s.context_buffer_size == 131072  # 128K
        assert s.enable_conversation_memory is True

    def test_document_processing_defaults(self):
        """Test document processing has sensible defaults."""
        s = settings

        assert s.chunk_size == 512
        assert s.chunk_overlap == 50
        assert s.enable_document_caching is True
        assert s.max_document_size_mb == 100

    def test_retrieval_defaults(self):
        """Test retrieval configuration defaults."""
        s = settings

        assert s.retrieval_strategy == "hybrid"  # Hybrid is optimal
        assert s.top_k == 10
        assert s.use_reranking is True  # BGE reranking enabled
        assert s.reranking_top_k == 5

    def test_embedding_defaults(self):
        """Test embedding configuration defaults."""
        s = settings

        assert s.embedding_model == "BAAI/bge-large-en-v1.5"
        assert s.embedding_dimension == 1024  # BGE-Large dimension
        assert s.use_sparse_embeddings is True  # SPLADE++ enabled

    def test_vector_database_defaults(self):
        """Test vector database configuration defaults."""
        s = settings

        assert s.vector_store_type == "qdrant"  # Qdrant is primary
        assert s.qdrant_url == "http://localhost:6333"
        assert s.qdrant_collection == "docmind_docs"

    def test_performance_defaults(self):
        """Test performance configuration defaults are reasonable."""
        s = settings

        assert s.max_query_latency_ms == 2000  # 2s max latency
        assert s.max_memory_gb == 4.0  # 4GB RAM limit
        assert s.max_vram_gb == 14.0  # 14GB for FP8 on RTX 4090
        assert s.enable_gpu_acceleration is True

    def test_vllm_defaults(self):
        """Test vLLM-specific defaults are optimized for RTX 4090."""
        s = settings

        assert s.vllm_gpu_memory_utilization == 0.85  # 85% utilization
        assert s.vllm_attention_backend == "FLASHINFER"  # FlashInfer backend
        assert s.vllm_enable_chunked_prefill is True
        assert s.vllm_max_num_batched_tokens == 8192
        assert s.vllm_max_num_seqs == 16

    def test_persistence_defaults(self):
        """Test persistence configuration creates proper paths."""
        s = settings

        assert s.data_dir == Path("./data")
        assert s.cache_dir == Path("./cache")
        assert s.sqlite_db_path == Path("./data/docmind.db")
        assert s.enable_wal_mode is True  # WAL mode for performance

    def test_centralized_constants_defaults(self):
        """Test all centralized constants have proper defaults."""
        s = settings

        # Memory conversion constants
        assert s.bytes_to_gb_divisor == 1024**3
        assert s.bytes_to_mb_divisor == 1024 * 1024

        # BGE-M3 constants
        assert s.bge_m3_embedding_dim == 1024
        assert s.bge_m3_max_length == 8192
        assert s.bge_m3_model_name == "BAAI/bge-m3"
        assert s.bge_m3_batch_size_gpu == 12
        assert s.bge_m3_batch_size_cpu == 4

        # Hybrid retrieval constants
        assert s.rrf_fusion_alpha == 60  # RRF fusion alpha parameter
        assert s.rrf_k_constant == 60
        assert s.rrf_fusion_weight_dense == 0.7
        assert s.rrf_fusion_weight_sparse == 0.3

        # App.py constants (moved from scattered files)
        assert s.default_token_limit == 131072  # 128K
        assert s.context_size_options == [8192, 32768, 65536, 131072]
        assert s.request_timeout_seconds == 60.0
        assert s.streaming_delay_seconds == 0.02
        assert s.minimum_vram_high_gb == 16
        assert s.minimum_vram_medium_gb == 8


class TestFieldValidation:
    """Test all field validators work correctly and catch invalid values."""

    def test_agent_timeout_range_validation(self):
        """Test agent decision timeout must be within valid range."""
        # Valid range
        DocMindSettings(agent_decision_timeout=100)
        DocMindSettings(agent_decision_timeout=500)
        DocMindSettings(agent_decision_timeout=1000)

        # Invalid: too low
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 100"
        ):
            DocMindSettings(agent_decision_timeout=50)

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 1000"
        ):
            DocMindSettings(agent_decision_timeout=2000)

    def test_max_agent_retries_validation(self):
        """Test max agent retries has reasonable bounds."""
        # Valid range
        DocMindSettings(max_agent_retries=0)
        DocMindSettings(max_agent_retries=3)
        DocMindSettings(max_agent_retries=5)

        # Invalid: negative
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            DocMindSettings(max_agent_retries=-1)

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 5"
        ):
            DocMindSettings(max_agent_retries=10)

    def test_llm_temperature_validation(self):
        """Test LLM temperature must be in valid range."""
        # Valid temperatures
        DocMindSettings(llm_temperature=0.0)
        DocMindSettings(llm_temperature=0.5)
        DocMindSettings(llm_temperature=2.0)

        # Invalid: negative
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            DocMindSettings(llm_temperature=-0.1)

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 2"
        ):
            DocMindSettings(llm_temperature=3.0)

    def test_llm_max_tokens_validation(self):
        """Test LLM max tokens has reasonable bounds."""
        # Valid token counts
        DocMindSettings(llm_max_tokens=128)
        DocMindSettings(llm_max_tokens=2048)
        DocMindSettings(llm_max_tokens=8192)

        # Invalid: too low
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 128"
        ):
            DocMindSettings(llm_max_tokens=64)

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 8192"
        ):
            DocMindSettings(llm_max_tokens=16384)

    def test_chunk_size_validation(self):
        """Test chunk size has reasonable bounds."""
        # Valid chunk sizes
        DocMindSettings(chunk_size=128)
        DocMindSettings(chunk_size=512)
        DocMindSettings(chunk_size=2048)

        # Invalid: too small
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 128"
        ):
            DocMindSettings(chunk_size=64)

        # Invalid: too large
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 2048"
        ):
            DocMindSettings(chunk_size=4096)

    def test_chunk_overlap_validation(self):
        """Test chunk overlap has reasonable bounds."""
        # Valid overlaps
        DocMindSettings(chunk_overlap=0)
        DocMindSettings(chunk_overlap=100)
        DocMindSettings(chunk_overlap=200)

        # Invalid: negative
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            DocMindSettings(chunk_overlap=-10)

        # Invalid: too large
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 200"
        ):
            DocMindSettings(chunk_overlap=500)

    def test_top_k_validation(self):
        """Test top_k retrieval has reasonable bounds."""
        # Valid top-k values
        DocMindSettings(top_k=1)
        DocMindSettings(top_k=10)
        DocMindSettings(top_k=50)

        # Invalid: zero or negative
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 1"
        ):
            DocMindSettings(top_k=0)

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 50"
        ):
            DocMindSettings(top_k=100)

    def test_bge_m3_dimension_validation(self):
        """Test BGE-M3 embedding dimension validation."""
        # Valid dimensions
        DocMindSettings(bge_m3_embedding_dim=512)
        DocMindSettings(bge_m3_embedding_dim=1024)
        DocMindSettings(bge_m3_embedding_dim=4096)

        # Invalid: too small
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 512"
        ):
            DocMindSettings(bge_m3_embedding_dim=256)

        # Invalid: too large
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 4096"
        ):
            DocMindSettings(bge_m3_embedding_dim=8192)

    def test_rrf_alpha_validation(self):
        """Test RRF alpha weight validation."""
        # Valid alpha values (integer range 10-100)
        DocMindSettings(rrf_fusion_alpha=10)
        DocMindSettings(rrf_fusion_alpha=60)  # Default
        DocMindSettings(rrf_fusion_alpha=100)

        # Invalid: too low
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 10"
        ):
            DocMindSettings(rrf_fusion_alpha=5)

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 100"
        ):
            DocMindSettings(rrf_fusion_alpha=150)

    def test_vllm_gpu_memory_validation(self):
        """Test vLLM GPU memory utilization validation."""
        # Valid utilization values
        DocMindSettings(vllm_gpu_memory_utilization=0.1)
        DocMindSettings(vllm_gpu_memory_utilization=0.85)
        DocMindSettings(vllm_gpu_memory_utilization=0.95)

        # Invalid: too low
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0.1"
        ):
            DocMindSettings(vllm_gpu_memory_utilization=0.05)

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 0.95"
        ):
            DocMindSettings(vllm_gpu_memory_utilization=1.0)

    def test_streamlit_port_validation(self):
        """Test Streamlit port validation."""
        # Valid ports
        DocMindSettings(streamlit_port=1024)
        DocMindSettings(streamlit_port=8501)
        DocMindSettings(streamlit_port=65535)

        # Invalid: too low (system ports)
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 1024"
        ):
            DocMindSettings(streamlit_port=80)

        # Invalid: too high
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 65535"
        ):
            DocMindSettings(streamlit_port=70000)


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
        DocMindSettings(sqlite_db_path=str(test_db_path))
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

    def test_openai_backend_warning(self):
        """Test OpenAI backend triggers local-compatible warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DocMindSettings(llm_backend="openai")

            # Should have exactly one warning
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "OpenAI backend selected" in str(w[0].message)
            assert "local-compatible endpoint" in str(w[0].message)

    def test_non_openai_backends_no_warning(self):
        """Test non-OpenAI backends don't trigger warnings."""
        for backend in ["ollama", "llamacpp", "vllm"]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                DocMindSettings(llm_backend=backend)
                assert len(w) == 0


class TestEnvironmentVariableOverrides:
    """Test environment variable overrides work correctly."""

    def test_basic_env_override(self):
        """Test basic environment variable override."""
        with patch.dict(os.environ, {"DOCMIND_DEBUG": "true"}):
            s = settings
            assert s.debug is True

    def test_numeric_env_override(self):
        """Test numeric environment variables are properly converted."""
        with patch.dict(
            os.environ,
            {
                "DOCMIND_AGENT_DECISION_TIMEOUT": "500",
                "DOCMIND_LLM_TEMPERATURE": "0.5",
                "DOCMIND_TOP_K": "15",
            },
        ):
            s = settings
            assert s.agent_decision_timeout == 500
            assert s.llm_temperature == 0.5
            assert s.top_k == 15

    def test_boolean_env_override(self):
        """Test boolean environment variables work correctly."""
        with patch.dict(
            os.environ,
            {
                "DOCMIND_ENABLE_MULTI_AGENT": "false",
                "DOCMIND_USE_RERANKING": "true",
                "DOCMIND_ENABLE_GPU_ACCELERATION": "false",
            },
        ):
            s = settings
            assert s.enable_multi_agent is False
            assert s.use_reranking is True
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
            s = settings
            assert s.data_dir == Path("/tmp/custom/data")
            assert s.cache_dir == Path("/tmp/custom/cache")

    def test_list_env_override(self):
        """Test list environment variables are handled correctly."""
        # Note: Pydantic typically expects JSON for list env vars
        with patch.dict(
            os.environ,
            {"DOCMIND_ANALYSIS_MODES": '["quick", "detailed", "comprehensive"]'},
        ):
            s = settings
            assert s.analysis_modes == ["quick", "detailed", "comprehensive"]

    def test_env_validation_still_applies(self):
        """Test validation still applies with environment overrides."""
        with (
            patch.dict(os.environ, {"DOCMIND_AGENT_DECISION_TIMEOUT": "5000"}),
            pytest.raises(
                ValidationError, match="Input should be less than or equal to 1000"
            ),
        ):
            _ = settings  # Trigger validation

    def test_env_prefix_enforced(self):
        """Test environment variables must have DOCMIND_ prefix."""
        with patch.dict(os.environ, {"DEBUG": "true"}):  # No prefix
            s = settings
            assert s.debug is False  # Should use default, not env var

    def test_case_insensitive_env_vars(self):
        """Test environment variables are case insensitive."""
        with patch.dict(os.environ, {"docmind_debug": "true"}):
            s = settings
            assert s.debug is True


class TestConfigurationMethods:
    """Test configuration method helpers return correct subsets."""

    def test_get_agent_config(self):
        """Test get_agent_config returns correct agent-specific settings."""
        s = settings
        agent_config = s.get_agent_config()

        expected_keys = {
            "enable_multi_agent",
            "agent_decision_timeout",
            "enable_fallback_rag",
            "max_agent_retries",
            "llm_backend",
            "model_name",
            "context_window_size",
            "context_buffer_size",
            "quantization",
            "kv_cache_dtype",
        }

        assert set(agent_config.keys()) == expected_keys
        assert agent_config["enable_multi_agent"] is True
        assert agent_config["agent_decision_timeout"] == 300
        assert agent_config["model_name"] == "Qwen/Qwen3-4B-Instruct-2507"

    def test_get_performance_config(self):
        """Test get_performance_config returns performance settings."""
        s = settings
        perf_config = s.get_performance_config()

        expected_keys = {
            "max_query_latency_ms",
            "agent_decision_timeout",
            "max_memory_gb",
            "max_vram_gb",
            "enable_gpu_acceleration",
            "enable_performance_logging",
            "vllm_gpu_memory_utilization",
            "vllm_attention_backend",
            "vllm_enable_chunked_prefill",
            "vllm_max_num_batched_tokens",
            "vllm_max_num_seqs",
        }

        assert set(perf_config.keys()) == expected_keys
        assert perf_config["max_vram_gb"] == 14.0
        assert perf_config["vllm_attention_backend"] == "FLASHINFER"

    def test_get_vllm_config(self):
        """Test get_vllm_config returns vLLM-specific settings."""
        s = settings
        vllm_config = s.get_vllm_config()

        expected_keys = {
            "model_name",
            "quantization",
            "kv_cache_dtype",
            "max_model_len",
            "gpu_memory_utilization",
            "attention_backend",
            "enable_chunked_prefill",
            "max_num_batched_tokens",
            "max_num_seqs",
            "default_temperature",
            "default_max_tokens",
        }

        assert set(vllm_config.keys()) == expected_keys
        assert vllm_config["quantization"] == "fp8"
        assert vllm_config["max_model_len"] == 131072  # 128K context
        assert vllm_config["attention_backend"] == "FLASHINFER"

    def test_to_dict_method(self):
        """Test to_dict method returns complete settings."""
        s = settings
        settings_dict = s.model_dump()

        # Should contain all fields
        assert "app_name" in settings_dict
        assert "enable_multi_agent" in settings_dict
        assert "model_name" in settings_dict
        assert "bge_m3_embedding_dim" in settings_dict

        # Values should match
        assert settings_dict["app_name"] == "DocMind AI"
        assert settings_dict["enable_multi_agent"] is True
        assert settings_dict["model_name"] == "Qwen/Qwen3-4B-Instruct-2507"


class TestCentralizedConstants:
    """Test all centralized constants are accessible and have correct values."""

    def test_memory_conversion_constants(self):
        """Test memory conversion constants are correct."""
        s = settings

        assert s.bytes_to_gb_divisor == 1024**3  # 1073741824
        assert s.bytes_to_mb_divisor == 1024 * 1024  # 1048576

    def test_bge_m3_constants(self):
        """Test BGE-M3 model constants are properly centralized."""
        s = settings

        assert s.bge_m3_embedding_dim == 1024
        assert s.bge_m3_max_length == 8192
        assert s.bge_m3_model_name == "BAAI/bge-m3"
        assert s.bge_m3_batch_size_gpu == 12  # Optimized for GPU
        assert s.bge_m3_batch_size_cpu == 4  # Conservative for CPU

    def test_hybrid_retrieval_constants(self):
        """Test hybrid retrieval constants are research-backed."""
        s = settings

        # RRF parameters
        assert s.rrf_fusion_alpha == 60  # RRF fusion alpha parameter
        assert s.rrf_k_constant == 60

        # Research-backed weights
        assert s.rrf_fusion_weight_dense == 0.7
        assert s.rrf_fusion_weight_sparse == 0.3

        # Weights should sum to 1.0
        assert s.rrf_fusion_weight_dense + s.rrf_fusion_weight_sparse == 1.0

    def test_default_processing_constants(self):
        """Test default processing value constants."""
        s = settings

        assert s.default_batch_size == 20
        assert s.default_confidence_threshold == 0.8
        assert s.default_entity_confidence == 0.8
        assert s.top_k == 10
        assert s.reranking_top_k == 5

    def test_timeout_configuration_constants(self):
        """Test timeout configuration constants."""
        s = settings

        assert s.default_qdrant_timeout == 60  # 1 minute
        assert s.default_agent_timeout == 3.0  # 3 seconds
        assert s.cache_expiry_seconds == 3600  # 1 hour
        assert s.spacy_download_timeout == 300  # 5 minutes

    def test_app_constants_moved_from_scattered_files(self):
        """Test constants moved from app.py and other scattered files."""
        s = settings

        # Token and context constants
        assert s.default_token_limit == 131072  # 128K
        assert s.context_size_options == [8192, 32768, 65536, 131072]
        assert s.suggested_context_high == 65536  # 64K
        assert s.suggested_context_medium == 32768  # 32K
        assert s.suggested_context_low == 8192  # 8K

        # Performance constants
        assert s.request_timeout_seconds == 60.0
        assert s.streaming_delay_seconds == 0.02
        assert s.minimum_vram_high_gb == 16
        assert s.minimum_vram_medium_gb == 8

    def test_monitoring_constants(self):
        """Test performance monitoring constants."""
        s = settings

        assert s.cpu_monitoring_interval == 0.1  # 100ms
        assert s.percent_multiplier == 100


class TestGlobalSettingsInstance:
    """Test the global settings instance works correctly."""

    def test_global_settings_instance_exists(self):
        """Test global settings instance is available."""
        assert settings is not None
        assert isinstance(settings, DocMindSettings)

    def test_global_settings_has_defaults(self):
        """Test global settings instance has proper defaults."""
        assert settings.app_name == "DocMind AI"
        assert settings.enable_multi_agent is True
        assert settings.model_name == "Qwen/Qwen3-4B-Instruct-2507"

    def test_global_settings_is_singleton(self):
        """Test global settings behaves like singleton."""
        from src.config.settings import settings as settings1
        from src.config.settings import settings as settings2

        # Should be the same instance
        assert settings1 is settings2


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_none_values_handled_correctly(self):
        """Test None values are handled correctly where allowed."""
        s = DocMindSettings(
            llm_api_key=None,  # Should be allowed
            log_file=None,  # Should be allowed
        )

        assert s.llm_api_key is None
        assert s.log_file is None

    def test_empty_string_handling(self):
        """Test empty strings are handled appropriately."""
        # Critical fields should reject empty strings
        with pytest.raises(
            ValidationError, match="Field cannot be empty or whitespace-only"
        ):
            DocMindSettings(app_name="")

        # These fields don't exist in the current DocMindSettings model
        # Testing validation with bge_m3_model_name instead
        with pytest.raises(
            ValidationError, match="String should have at least 1 character"
        ):
            DocMindSettings(bge_m3_model_name="")

        # Test whitespace-only strings are also rejected
        with pytest.raises(
            ValidationError, match="Field cannot be empty or whitespace-only"
        ):
            DocMindSettings(app_name="   ")

        with pytest.raises(
            ValidationError, match="String should have at least 1 character"
        ):
            DocMindSettings(bge_m3_model_name="  \t  ")

    def test_extreme_boundary_values(self):
        """Test extreme boundary values."""
        # Test minimum values
        s = DocMindSettings(
            agent_decision_timeout=100,  # Minimum allowed
            max_agent_retries=0,  # Minimum allowed
            llm_temperature=0.0,  # Minimum allowed
            chunk_overlap=0,  # Minimum allowed
        )

        assert s.agent_decision_timeout == 100
        assert s.max_agent_retries == 0
        assert s.llm_temperature == 0.0
        assert s.chunk_overlap == 0

        # Test maximum values
        s = DocMindSettings(
            agent_decision_timeout=1000,  # Maximum allowed
            max_agent_retries=5,  # Maximum allowed
            llm_temperature=2.0,  # Maximum allowed
            top_k=50,  # Maximum allowed
        )

        assert s.agent_decision_timeout == 1000
        assert s.max_agent_retries == 5
        assert s.llm_temperature == 2.0
        assert s.top_k == 50

    def test_type_coercion_edge_cases(self):
        """Test type coercion works correctly."""
        # String numbers should be converted
        with patch.dict(
            os.environ, {"DOCMIND_TOP_K": "15", "DOCMIND_LLM_TEMPERATURE": "0.5"}
        ):
            s = settings
            assert isinstance(s.top_k, int)
            assert isinstance(s.llm_temperature, float)
            assert s.top_k == 15
            assert s.llm_temperature == 0.5

    def test_invalid_literal_values(self):
        """Test invalid literal values are rejected."""
        with pytest.raises(ValidationError, match="Input should be"):
            DocMindSettings(llm_backend="invalid_backend")

        with pytest.raises(ValidationError, match="Input should be"):
            DocMindSettings(retrieval_strategy="invalid_strategy")

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


class TestBackwardCompatibilityAndIntegration:
    """Test backward compatibility with existing code that imports constants."""

    def test_constants_accessible_via_settings_instance(self):
        """Test all moved constants are accessible via settings."""
        s = settings

        # These constants were moved from scattered files
        # and should be accessible via the centralized settings
        constants_to_test = [
            "bytes_to_gb_divisor",
            "bge_m3_embedding_dim",
            "rrf_fusion_alpha",
            "default_batch_size",
            "default_token_limit",
            "request_timeout_seconds",
            "minimum_vram_high_gb",
        ]

        for constant in constants_to_test:
            assert hasattr(s, constant)
            value = getattr(s, constant)
            assert value is not None
            # All these constants should be positive numbers
            assert isinstance(value, int | float)
            assert value > 0

    def test_settings_integrates_with_existing_modules(self):
        """Test settings can be imported and used by existing modules."""
        # This simulates how existing modules import settings

        # Common usage patterns that existing modules should support
        agent_config = settings.get_agent_config()
        perf_config = settings.get_performance_config()
        vllm_config = settings.get_vllm_config()

        # All configs should be dicts with expected keys
        assert isinstance(agent_config, dict)
        assert isinstance(perf_config, dict)
        assert isinstance(vllm_config, dict)

        assert len(agent_config) > 0
        assert len(perf_config) > 0
        assert len(vllm_config) > 0

    def test_no_import_errors_from_settings(self):
        """Test importing settings doesn't cause circular imports or other issues."""
        # This should not raise any exceptions
        try:
            # Basic usage should work
            s = settings
            config = s.model_dump()

            assert isinstance(config, dict)
            assert len(config) > 50  # Should have many settings

        except ImportError as e:
            pytest.fail(f"Failed to import settings: {e}")


class TestRealWorldScenarios:
    """Test real-world usage scenarios and business logic."""

    def test_rtx_4090_optimized_configuration(self):
        """Test settings are optimized for RTX 4090 16GB setup."""
        s = settings

        # Memory settings should be optimized for 16GB VRAM
        assert s.max_vram_gb == 14.0  # Leave 2GB headroom
        assert s.vllm_gpu_memory_utilization == 0.85  # Conservative

        # FP8 optimization should be enabled
        assert s.quantization == "fp8"
        assert s.kv_cache_dtype == "fp8"
        assert s.enable_kv_cache_optimization is True

        # FlashInfer should be configured
        assert s.vllm_attention_backend == "FLASHINFER"

    def test_128k_context_configuration(self):
        """Test 128K context window is properly configured."""
        s = settings

        assert s.context_window_size == 131072  # 128K
        assert s.context_buffer_size == 131072
        assert s.default_token_limit == 131072

        # Context options should include 128K
        assert 131072 in s.context_size_options

    def test_local_first_configuration(self):
        """Test configuration enforces local-first architecture."""
        s = settings

        # Default backend should be local
        assert s.llm_backend in ["ollama", "llamacpp", "vllm"]  # Not OpenAI
        assert s.llm_base_url.startswith("http://localhost")
        assert s.qdrant_url.startswith("http://localhost")
        assert s.llm_api_key is None  # No API key needed

    def test_hybrid_search_configuration(self):
        """Test hybrid search is properly configured."""
        s = settings

        assert s.retrieval_strategy == "hybrid"
        assert s.use_sparse_embeddings is True
        assert s.use_reranking is True

        # Research-backed weights should be configured
        assert s.rrf_fusion_weight_dense == 0.7
        assert s.rrf_fusion_weight_sparse == 0.3

    def test_production_ready_defaults(self):
        """Test defaults are suitable for production use."""
        s = settings

        # Performance settings should be reasonable
        assert s.max_query_latency_ms <= 5000  # Under 5 seconds
        assert s.max_memory_gb <= 8.0  # Reasonable RAM usage
        assert s.agent_decision_timeout <= 1000  # Under 1 second

        # Caching should be enabled for performance
        assert s.enable_document_caching is True
        assert s.cache_expiry_seconds > 0

        # Multi-agent should be enabled by default
        assert s.enable_multi_agent is True
        assert s.enable_fallback_rag is True

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
        assert dev_settings.model_name == prod_settings.model_name
        assert dev_settings.enable_multi_agent == prod_settings.enable_multi_agent
