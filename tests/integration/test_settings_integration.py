"""Integration tests for the centralized settings system.

This module tests how the centralized settings system integrates with other
modules and components throughout the DocMind AI application. These tests
validate that settings work correctly in real-world cross-module scenarios.

Integration tests cover:
- Settings usage across different modules
- Configuration methods integration with actual components
- Backward compatibility with modules importing from old settings
- Cross-module consistency of settings usage
- Directory creation integration with file operations
- LLM backend integration scenarios
- vLLM configuration integration with actual vLLM usage patterns
- Agent coordination integration scenarios

These tests use lightweight models and mocking to avoid full system overhead
while still validating integration points.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import DocMindSettings


class TestCrossModuleIntegration:
    """Test settings integration across different modules."""

    def test_app_module_settings_integration(self):
        """Test app.py module can properly use centralized settings."""
        s = DocMindSettings()

        # Test constants that were moved from app.py
        assert hasattr(s, "default_token_limit")
        assert hasattr(s, "context_size_options")
        assert hasattr(s, "request_timeout_seconds")
        assert hasattr(s, "streaming_delay_seconds")

        # These should be valid for Streamlit app usage
        assert s.default_token_limit >= 8192  # Minimum useful context
        assert len(s.context_size_options) > 0
        assert all(isinstance(size, int) for size in s.context_size_options)
        assert s.request_timeout_seconds > 0
        assert s.streaming_delay_seconds > 0

    def test_tool_factory_settings_integration(self):
        """Test tool factory can access all needed settings."""
        s = DocMindSettings()

        # Tool factory needs agent configuration
        agent_config = s.get_agent_config()

        # Should contain keys needed for tool creation
        required_keys = [
            "enable_multi_agent",
            "agent_decision_timeout",
            "max_agent_retries",
            "llm_backend",
            "model_name",
        ]

        for key in required_keys:
            assert key in agent_config
            assert agent_config[key] is not None

    def test_retrieval_module_settings_integration(self):
        """Test retrieval modules can access BGE-M3 and hybrid search settings."""
        s = DocMindSettings()

        # BGE-M3 constants should be accessible
        assert hasattr(s, "bge_m3_embedding_dim")
        assert hasattr(s, "bge_m3_max_length")
        assert hasattr(s, "bge_m3_batch_size_gpu")
        assert hasattr(s, "bge_m3_batch_size_cpu")

        # Hybrid retrieval constants
        assert hasattr(s, "rrf_fusion_alpha")
        assert hasattr(s, "rrf_fusion_weight_dense")
        assert hasattr(s, "rrf_fusion_weight_sparse")

        # Values should be suitable for retrieval operations
        assert 512 <= s.bge_m3_embedding_dim <= 4096
        assert 1024 <= s.bge_m3_max_length <= 16384
        assert 10 <= s.rrf_fusion_alpha <= 100

    def test_vllm_module_settings_integration(self):
        """Test vLLM configuration integrates with actual vLLM usage patterns."""
        s = DocMindSettings()

        vllm_config = s.get_vllm_config()

        # vLLM config should have all required parameters
        required_vllm_keys = [
            "model_name",
            "quantization",
            "kv_cache_dtype",
            "max_model_len",
            "gpu_memory_utilization",
            "attention_backend",
            "enable_chunked_prefill",
            "max_num_batched_tokens",
            "max_num_seqs",
        ]

        for key in required_vllm_keys:
            assert key in vllm_config
            assert vllm_config[key] is not None

        # Values should be valid for vLLM
        assert vllm_config["quantization"] in ["fp8", "int8", "int4", "awq"]
        assert vllm_config["attention_backend"] in ["FLASHINFER"]
        assert 0.1 <= vllm_config["gpu_memory_utilization"] <= 0.95
        assert vllm_config["max_model_len"] >= 8192  # Reasonable context window

    def test_coordinator_settings_integration(self):
        """Test multi-agent coordinator can use settings properly."""
        s = DocMindSettings()

        # Coordinator needs both agent and performance config
        agent_config = s.get_agent_config()
        perf_config = s.get_performance_config()

        # Should have coordination settings
        assert agent_config["enable_multi_agent"] is not None
        assert agent_config["agent_decision_timeout"] > 0
        assert agent_config["max_agent_retries"] >= 0

        # Performance settings for coordination
        assert perf_config["max_query_latency_ms"] > 0
        assert perf_config["max_memory_gb"] > 0
        assert perf_config["max_vram_gb"] > 0

    def test_database_utils_settings_integration(self):
        """Test database utilities can access persistence settings."""
        s = DocMindSettings()

        # Database settings should be accessible
        assert hasattr(s, "sqlite_db_path")
        assert hasattr(s, "enable_wal_mode")
        assert hasattr(s, "data_dir")

        # Values should be valid for database operations
        assert isinstance(s.sqlite_db_path, Path)
        assert isinstance(s.enable_wal_mode, bool)
        assert isinstance(s.data_dir, Path)

    def test_monitoring_utils_settings_integration(self):
        """Test monitoring utilities can access performance monitoring settings."""
        s = DocMindSettings()

        # Monitoring constants should be accessible
        assert hasattr(s, "cpu_monitoring_interval")
        assert hasattr(s, "percent_multiplier")
        assert hasattr(s, "enable_performance_logging")

        # Values should be suitable for monitoring
        assert 0.01 <= s.cpu_monitoring_interval <= 1.0
        assert s.percent_multiplier == 100
        assert isinstance(s.enable_performance_logging, bool)


class TestSettingsWithActualFileOperations:
    """Test settings integration with actual file system operations."""

    def test_directory_creation_with_actual_operations(self, tmp_path):
        """Test directory creation works with actual file operations."""
        # Use temporary directory for testing
        test_data_dir = tmp_path / "integration_data"
        test_cache_dir = tmp_path / "integration_cache"
        test_log_file = tmp_path / "logs" / "integration.log"

        # Create settings with custom directories
        s = DocMindSettings(
            data_dir=str(test_data_dir),
            cache_dir=str(test_cache_dir),
            log_file=str(test_log_file),
        )

        # Settings object should exist
        assert s is not None

        # Directories should be created
        assert test_data_dir.exists()
        assert test_cache_dir.exists()
        assert test_log_file.parent.exists()

        # Should be able to create files in these directories
        test_file = test_data_dir / "test_document.txt"
        test_file.write_text("Test document content")
        assert test_file.exists()

        cache_file = test_cache_dir / "test_cache.json"
        cache_file.write_text('{"cached": true}')
        assert cache_file.exists()

    def test_sqlite_database_path_integration(self, tmp_path):
        """Test SQLite database path works with actual database operations."""
        test_db_path = tmp_path / "db" / "test_integration.db"

        s = DocMindSettings(sqlite_db_path=str(test_db_path))

        # Parent directory should be created
        assert test_db_path.parent.exists()

        # Should be able to create database file
        import sqlite3

        # This simulates actual database usage
        with sqlite3.connect(str(s.sqlite_db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)"
            )
            cursor.execute("INSERT INTO test_table (name) VALUES (?)", ("test",))
            conn.commit()

            # Verify the data
            cursor.execute("SELECT name FROM test_table WHERE id = 1")
            result = cursor.fetchone()
            assert result[0] == "test"

    def test_nested_directory_creation_integration(self, tmp_path):
        """Test deeply nested directory creation works in practice."""
        nested_path = tmp_path / "level1" / "level2" / "level3" / "level4"

        s = DocMindSettings(data_dir=str(nested_path))

        # Settings object should exist
        assert s is not None

        # All levels should be created
        assert nested_path.exists()
        assert nested_path.is_dir()

        # Should be able to create files at any level
        for i, parent in enumerate([nested_path] + list(nested_path.parents)):
            if parent.name.startswith("level"):
                test_file = parent / f"test_file_{i}.txt"
                test_file.write_text(f"Content {i}")
                assert test_file.exists()


class TestEnvironmentIntegration:
    """Test settings integration with environment variables in realistic scenarios."""

    def test_production_environment_simulation(self):
        """Test settings work in simulated production environment."""
        production_env = {
            "DOCMIND_DEBUG": "false",
            "DOCMIND_LOG_LEVEL": "INFO",
            "DOCMIND_ENABLE_PERFORMANCE_LOGGING": "true",
            "DOCMIND_MAX_MEMORY_GB": "8.0",
            "DOCMIND_MAX_VRAM_GB": "14.0",
            "DOCMIND_VLLM_GPU_MEMORY_UTILIZATION": "0.85",
            "DOCMIND_ENABLE_DOCUMENT_CACHING": "true",
        }

        with patch.dict("os.environ", production_env):
            s = DocMindSettings()

            # Production settings should be applied
            assert s.debug is False
            assert s.log_level == "INFO"
            assert s.enable_performance_logging is True
            assert s.max_memory_gb == 8.0
            assert s.max_vram_gb == 14.0
            assert s.vllm_gpu_memory_utilization == 0.85
            assert s.enable_document_caching is True

    def test_development_environment_simulation(self):
        """Test settings work in simulated development environment."""
        dev_env = {
            "DOCMIND_DEBUG": "true",
            "DOCMIND_LOG_LEVEL": "DEBUG",
            "DOCMIND_ENABLE_PERFORMANCE_LOGGING": "false",
            "DOCMIND_AGENT_DECISION_TIMEOUT": "200",  # Faster for dev
            "DOCMIND_LLM_TEMPERATURE": "0.3",  # Slightly higher for testing
        }

        with patch.dict("os.environ", dev_env):
            s = DocMindSettings()

            # Development settings should be applied
            assert s.debug is True
            assert s.log_level == "DEBUG"
            assert s.enable_performance_logging is False
            assert s.agent_decision_timeout == 200
            assert s.llm_temperature == 0.3

    def test_gpu_configuration_environment_simulation(self):
        """Test GPU-specific environment configuration."""
        gpu_env = {
            "DOCMIND_ENABLE_GPU_ACCELERATION": "true",
            "DOCMIND_VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "DOCMIND_QUANTIZATION": "fp8",
            "DOCMIND_KV_CACHE_DTYPE": "fp8",
            "DOCMIND_VLLM_ENABLE_CHUNKED_PREFILL": "true",
        }

        with patch.dict("os.environ", gpu_env):
            s = DocMindSettings()

            # GPU settings should be optimized
            assert s.enable_gpu_acceleration is True
            assert s.vllm_attention_backend == "FLASHINFER"
            assert s.quantization == "fp8"
            assert s.kv_cache_dtype == "fp8"
            assert s.vllm_enable_chunked_prefill is True

    def test_hybrid_search_environment_configuration(self):
        """Test hybrid search environment configuration."""
        hybrid_env = {
            "DOCMIND_RETRIEVAL_STRATEGY": "hybrid",
            "DOCMIND_USE_SPARSE_EMBEDDINGS": "true",
            "DOCMIND_USE_RERANKING": "true",
            "DOCMIND_TOP_K": "15",
            "DOCMIND_RERANKER_TOP_K": "8",
        }

        with patch.dict("os.environ", hybrid_env):
            s = DocMindSettings()

            # Hybrid search should be configured
            assert s.retrieval_strategy == "hybrid"
            assert s.use_sparse_embeddings is True
            assert s.use_reranking is True
            assert s.top_k == 15
            assert s.reranker_top_k == 8


class TestBackwardCompatibilityIntegration:
    """Test backward compatibility with existing module imports."""

    def test_old_settings_still_work_alongside_new(self):
        """Test that old settings system still works while new system is active."""
        # This tests that both settings systems can coexist during migration

        # Both should be importable and instantiable
        new_settings = DocMindSettings()
        old_settings = DocMindSettings()

        # Both should have basic functionality
        assert hasattr(new_settings, "model_name")
        assert hasattr(old_settings, "model_name")

        # New settings should have more comprehensive configuration
        new_dict = new_settings.model_dump()
        old_dict = old_settings.model_dump()

        # New settings should have significantly more fields
        assert len(new_dict) > len(old_dict) * 2

    def test_modules_can_migrate_from_old_to_new_settings(self):
        """Test migration path from old to new settings imports."""
        # Simulate a module migrating from old to new settings

        # Old import pattern (still works)
        # New import pattern
        from src.config.settings import settings as new_settings
        from src.config.settings import settings as old_settings

        # Both should work
        assert old_settings is not None
        assert new_settings is not None

        # New settings should have more comprehensive configuration
        assert hasattr(new_settings, "enable_multi_agent")
        assert hasattr(new_settings, "vllm_attention_backend")
        assert hasattr(new_settings, "bge_m3_embedding_dim")

        # These might not be in old settings
        assert (
            not hasattr(old_settings, "enable_multi_agent")
            or old_settings.enable_multi_agent is not None
        )

    def test_re_exported_constants_accessibility(self):
        """Test that constants moved to centralized settings are still accessible."""
        s = DocMindSettings()

        # Constants that were moved from different modules should be accessible
        moved_constants = [
            "bytes_to_gb_divisor",  # From various utility modules
            "bge_m3_embedding_dim",  # From embedding modules
            "rrf_fusion_alpha",  # From retrieval modules
            "default_token_limit",  # From app.py
            "request_timeout_seconds",  # From app.py
            "minimum_vram_high_gb",  # From performance modules
        ]

        for constant in moved_constants:
            assert hasattr(s, constant)
            value = getattr(s, constant)
            assert value is not None
            assert isinstance(value, int | float)


class TestConfigurationMethodIntegration:
    """Test configuration methods work properly in integration scenarios."""

    @patch("src.agents.coordinator.MultiAgentCoordinator")
    def test_agent_config_integrates_with_coordinator(self, mock_coordinator):
        """Test agent configuration integrates with actual coordinator usage."""
        s = DocMindSettings()
        agent_config = s.get_agent_config()

        # Simulate coordinator creation with settings
        mock_coordinator.return_value = MagicMock()
        coordinator_instance = mock_coordinator(**agent_config)

        # Should be able to create coordinator with agent config
        assert coordinator_instance is not None
        mock_coordinator.assert_called_once_with(**agent_config)

    def test_performance_config_integrates_with_monitoring(self):
        """Test performance configuration integrates with monitoring systems."""
        s = DocMindSettings()
        perf_config = s.get_performance_config()

        # Performance config should have metrics suitable for monitoring
        assert perf_config["max_query_latency_ms"] > 0
        assert perf_config["max_memory_gb"] > 0
        assert perf_config["max_vram_gb"] > 0

        # Should be able to use these for monitoring thresholds
        memory_threshold = perf_config["max_memory_gb"] * 0.9  # 90% threshold
        vram_threshold = perf_config["max_vram_gb"] * 0.9

        assert memory_threshold > 0
        assert vram_threshold > 0

    def test_vllm_config_realistic_integration(self):
        """Test vLLM config works with realistic vLLM usage patterns."""
        s = DocMindSettings()
        vllm_config = s.get_vllm_config()

        # Simulate vLLM server configuration
        server_config = {
            "model": vllm_config["model_name"],
            "quantization": vllm_config["quantization"],
            "kv_cache_dtype": vllm_config["kv_cache_dtype"],
            "max_model_len": vllm_config["max_model_len"],
            "gpu_memory_utilization": vllm_config["gpu_memory_utilization"],
        }

        # Configuration should be valid for vLLM
        assert server_config["model"]  # Should have model name
        assert server_config["quantization"] in ["fp8", "int8", "int4", "awq"]
        assert server_config["max_model_len"] >= 8192
        assert 0.1 <= server_config["gpu_memory_utilization"] <= 0.95

    def test_to_dict_integration_with_serialization(self):
        """Test to_dict method integrates with actual serialization needs."""
        s = DocMindSettings()
        settings_dict = s.model_dump()

        # Should be JSON serializable (common integration need)
        import json

        try:
            json_str = json.dumps(
                settings_dict, default=str
            )  # default=str handles Path objects
            roundtrip_dict = json.loads(json_str)

            # Should be able to roundtrip
            assert isinstance(roundtrip_dict, dict)
            assert len(roundtrip_dict) > 50  # Should have many settings

        except (TypeError, ValueError) as e:
            pytest.fail(f"Settings dict not JSON serializable: {e}")


class TestRealWorldUsagePatterns:
    """Test settings in realistic application usage patterns."""

    def test_streamlit_app_integration_pattern(self):
        """Test settings work with typical Streamlit app usage patterns."""
        s = DocMindSettings()

        # Streamlit app would typically access these settings
        app_config = {
            "title": s.app_name,
            "version": s.app_version,
            "port": s.streamlit_port,
            "debug": s.debug,
            "context_options": s.context_size_options,
            "default_context": s.default_token_limit,
            "timeout": s.request_timeout_seconds,
            "streaming_delay": s.streaming_delay_seconds,
        }

        # All config values should be valid for Streamlit
        assert app_config["title"]  # Non-empty title
        assert app_config["version"]  # Version string
        assert 1024 <= app_config["port"] <= 65535  # Valid port
        assert isinstance(app_config["debug"], bool)
        assert len(app_config["context_options"]) > 0
        assert app_config["default_context"] in app_config["context_options"]
        assert app_config["timeout"] > 0
        assert app_config["streaming_delay"] > 0

    def test_document_processing_integration_pattern(self):
        """Test settings work with document processing workflows."""
        s = DocMindSettings()

        # Document processing would use these settings
        processing_config = {
            "chunk_size": s.chunk_size,
            "chunk_overlap": s.chunk_overlap,
            "max_doc_size_mb": s.max_document_size_mb,
            "enable_document_caching": s.enable_document_caching,
            "data_dir": s.data_dir,
            "cache_dir": s.cache_dir,
        }

        # Should be valid for document processing
        assert processing_config["chunk_size"] > processing_config["chunk_overlap"]
        assert processing_config["max_doc_size_mb"] > 0
        assert isinstance(processing_config["enable_document_caching"], bool)
        assert isinstance(processing_config["data_dir"], Path)
        assert isinstance(processing_config["cache_dir"], Path)

    def test_embedding_pipeline_integration_pattern(self):
        """Test settings work with embedding pipeline usage."""
        s = DocMindSettings()

        # Embedding pipeline would use these settings
        embedding_config = {
            "model_name": s.embedding_model,
            "dimension": s.embedding_dimension,
            "use_sparse": s.use_sparse_embeddings,
            "batch_size_gpu": s.bge_m3_batch_size_gpu,
            "batch_size_cpu": s.bge_m3_batch_size_cpu,
            "max_length": s.bge_m3_max_length,
        }

        # Should be valid for embedding operations
        assert embedding_config["model_name"]  # Should have model name
        assert embedding_config["dimension"] > 0
        assert isinstance(embedding_config["use_sparse"], bool)
        assert embedding_config["batch_size_gpu"] > 0
        assert embedding_config["batch_size_cpu"] > 0
        assert embedding_config["max_length"] >= 512

    def test_retrieval_system_integration_pattern(self):
        """Test settings work with retrieval system workflows."""
        s = DocMindSettings()

        # Retrieval system would use these settings
        retrieval_config = {
            "strategy": s.retrieval_strategy,
            "top_k": s.top_k,
            "reranker_top_k": s.reranker_top_k,
            "use_reranking": s.use_reranking,
            "rrf_alpha": s.rrf_fusion_alpha,
            "dense_weight": s.rrf_fusion_weight_dense,
            "sparse_weight": s.rrf_fusion_weight_sparse,
            "vector_store": s.vector_store_type,
            "qdrant_url": s.qdrant_url,
            "collection": s.qdrant_collection,
        }

        # Should be valid for retrieval operations
        assert retrieval_config["strategy"] in ["vector", "hybrid", "graphrag"]
        assert retrieval_config["top_k"] > retrieval_config["reranker_top_k"]
        assert isinstance(retrieval_config["use_reranking"], bool)
        assert 0 < retrieval_config["rrf_alpha"] < 1
        assert (
            retrieval_config["dense_weight"] + retrieval_config["sparse_weight"] == 1.0
        )
        assert retrieval_config["vector_store"] in ["qdrant", "chroma", "weaviate"]
        assert retrieval_config["qdrant_url"].startswith("http")
        assert retrieval_config["collection"]  # Non-empty collection name

    def test_multi_agent_coordination_integration_pattern(self):
        """Test settings work with multi-agent coordination workflows."""
        s = DocMindSettings()

        # Multi-agent system would use these settings
        coordination_config = {
            "enabled": s.enable_multi_agent,
            "timeout_ms": s.agent_decision_timeout,
            "max_retries": s.max_agent_retries,
            "fallback_enabled": s.enable_fallback_rag,
            "llm_backend": s.llm_backend,
            "model_name": s.model_name,
            "context_window": s.context_window_size,
            "performance_logging": s.enable_performance_logging,
        }

        # Should be valid for agent coordination
        assert isinstance(coordination_config["enabled"], bool)
        assert coordination_config["timeout_ms"] > 0
        assert coordination_config["max_retries"] >= 0
        assert isinstance(coordination_config["fallback_enabled"], bool)
        assert coordination_config["llm_backend"] in [
            "ollama",
            "llamacpp",
            "vllm",
            "openai",
        ]
        assert coordination_config["model_name"]  # Non-empty model name
        assert coordination_config["context_window"] >= 8192  # Reasonable context
        assert isinstance(coordination_config["performance_logging"], bool)


class TestSettingsWithMockedComponents:
    """Test settings integration with mocked external components."""

    @patch("qdrant_client.QdrantClient")
    def test_qdrant_integration_with_settings(self, mock_qdrant):
        """Test settings work properly with Qdrant client integration."""
        s = DocMindSettings()

        # Mock Qdrant client
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client

        # Simulate Qdrant client creation with settings
        client = mock_qdrant(url=s.qdrant_url, timeout=s.default_qdrant_timeout)

        # Should be able to create client with settings
        assert client is not None
        mock_qdrant.assert_called_once_with(
            url=s.qdrant_url, timeout=s.default_qdrant_timeout
        )

    @patch("sqlite3.connect")
    def test_sqlite_integration_with_settings(self, mock_sqlite):
        """Test settings work properly with SQLite database integration."""
        s = DocMindSettings()

        # Mock SQLite connection
        mock_conn = MagicMock()
        mock_sqlite.return_value = mock_conn

        # Simulate database connection with settings
        conn = mock_sqlite(str(s.sqlite_db_path))

        # Should be able to connect with settings path
        assert conn is not None
        mock_sqlite.assert_called_once_with(str(s.sqlite_db_path))

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_embedding_model_integration_with_settings(
        self, mock_tokenizer, mock_model
    ):
        """Test settings work with embedding model loading integration."""
        s = DocMindSettings()

        # Mock model and tokenizer
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        # Simulate model loading with settings
        tokenizer = mock_tokenizer.from_pretrained(s.embedding_model)
        model = mock_model.from_pretrained(s.embedding_model)

        # Should be able to load with settings model name
        assert tokenizer is not None
        assert model is not None
        mock_tokenizer.from_pretrained.assert_called_once_with(s.embedding_model)
        mock_model.from_pretrained.assert_called_once_with(s.embedding_model)

    def test_async_integration_patterns(self):
        """Test settings work with async integration patterns."""
        import asyncio

        async def async_operation_with_settings():
            s = DocMindSettings()

            # Simulate async operations using settings
            await asyncio.sleep(s.streaming_delay_seconds)  # Use settings delay

            # Return configuration that would be used in async context
            return {
                "timeout": s.default_agent_timeout,
                "max_retries": s.max_agent_retries,
                "context_window": s.context_window_size,
            }

        # Should work in async context
        result = asyncio.run(async_operation_with_settings())

        assert result["timeout"] > 0
        assert result["max_retries"] >= 0
        assert result["context_window"] >= 8192


class TestConcurrentSettingsAccess:
    """Test settings work correctly under concurrent access patterns."""

    def test_thread_safe_settings_access(self):
        """Test settings can be safely accessed from multiple threads."""
        import threading

        results = []

        def access_settings():
            s = DocMindSettings()
            # Access multiple settings to test thread safety
            config = {
                "model_name": s.model_name,
                "context_window": s.context_window_size,
                "batch_size": s.bge_m3_batch_size_gpu,
                "timeout": s.agent_decision_timeout,
            }
            results.append(config)

        # Create multiple threads accessing settings concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=access_settings)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All threads should get consistent results
        assert len(results) == 10
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result  # All should be identical

    def test_global_settings_singleton_behavior(self):
        """Test global settings instance behaves consistently."""
        # Import settings multiple times
        from src.config.settings import settings as s1
        from src.config.settings import settings as s2

        # Should be the same instance
        assert s1 is s2

        # Modifications should be reflected globally (though not recommended)
        original_debug = s1.debug

        # Even though settings is technically mutable, in practice it should be
        # treated as immutable. This test just verifies singleton behavior
        assert s2.debug == original_debug
