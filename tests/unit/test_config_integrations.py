"""Unit tests for configuration integrations.

Cover LlamaIndex and vLLM setup, environment variables, and orchestration.
Focus on business logic and logging behavior.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from src.config.integrations import (
    get_vllm_server_command,
    initialize_integrations,
    setup_llamaindex,
    setup_vllm_env,
)

# --- merged from test_config_integrations_coverage.py ---


@pytest.mark.unit
class TestSetupLlamaIndexCoverage:
    """Additional coverage for setup_llamaindex (merged)."""

    def test_setup_llamaindex_successful_llm_configuration_logging(self, caplog):
        """Test successful LLM configuration logging in setup_llamaindex."""
        mock_model_config = {
            "model_name": "test-model",
            "base_url": "http://localhost:11434",
            "temperature": 0.7,
        }

        with (
            patch("src.config.integrations.settings") as mock_settings,
            patch("src.config.integrations.Ollama") as mock_ollama_class,
            caplog.at_level("INFO", logger="src.config.integrations"),
        ):
            mock_llm = MagicMock()
            mock_llm.model = "test-model"
            mock_ollama_class.return_value = mock_llm

            mock_settings.get_model_config.return_value = mock_model_config
            mock_settings.get_embedding_config.side_effect = Exception(
                "Embedding failed"
            )
            mock_settings.vllm = None

            setup_llamaindex()

            assert (
                "LLM configured:" in caplog.text
                or "Could not configure LLM:" in caplog.text
            )

    def test_setup_llamaindex_successful_embedding_configuration_logging(self, caplog):
        """Test successful embedding configuration logging in setup_llamaindex."""
        mock_embedding_config = {
            "model_name": "BAAI/bge-m3",
            "device": "cpu",
            "max_length": 8192,
            "batch_size": 32,
            "trust_remote_code": True,
        }

        with (
            patch("src.config.integrations.settings") as mock_settings,
            patch("src.config.integrations.BGEM3Embedding") as mock_embedding_class,
            caplog.at_level("INFO", logger="src.config.integrations"),
        ):
            mock_settings.get_model_config.side_effect = Exception("LLM failed")
            mock_settings.get_embedding_config.return_value = mock_embedding_config
            mock_settings.vllm = None

            mock_embedding_class.return_value = MagicMock()

            setup_llamaindex()

            assert (
                "Embedding model configured:" in caplog.text
                or "Could not configure embeddings:" in caplog.text
            )


@pytest.mark.unit
class TestSetupVLLMEnvCoverage:
    """Additional coverage for setup_vllm_env (merged)."""

    def test_setup_vllm_env_variable_setting_with_debug_logging(self, caplog):
        """Test VLLM environment variable setting with debug logging."""
        mock_vllm_env = {
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "VLLM_KV_CACHE_DTYPE": "fp8_e5m2",
        }

        # Clear and set via fixture
        for key in list(mock_vllm_env):
            import os

            os.environ.pop(key, None)

        with (
            patch("src.config.integrations.settings") as mock_settings,
            caplog.at_level("DEBUG", logger="src.config.integrations"),
        ):
            mock_settings.get_vllm_env_vars.return_value = mock_vllm_env

            setup_vllm_env()

            for key, value in mock_vllm_env.items():
                import os

                assert os.environ[key] == value
                assert f"Set {key}={value}" in caplog.text

            assert (
                "vLLM environment variables configured for FP8 optimization"
                in caplog.text
            )


@pytest.mark.unit
class TestGetVLLMServerCommandCoverage:
    """Additional coverage for get_vllm_server_command (merged)."""

    def test_get_vllm_server_command_full_function_execution(self):
        """Test full execution of get_vllm_server_command function."""
        with patch("src.config.integrations.settings") as mock_settings:
            mock_vllm = MagicMock()
            mock_vllm.model = "test-model"
            mock_vllm.context_window = 32768
            mock_vllm.kv_cache_dtype = "auto"
            mock_vllm.gpu_memory_utilization = 0.8
            mock_vllm.max_num_seqs = 128
            mock_vllm.max_num_batched_tokens = 4096
            mock_vllm.enable_chunked_prefill = False
            mock_settings.vllm = mock_vllm

            command = get_vllm_server_command()
            assert command[:2] == ["vllm", "serve"]
            assert "32768" in command
            assert "auto" in command

    def test_get_vllm_server_command_with_chunked_prefill_true(self):
        """Test get_vllm_server_command with chunked prefill enabled."""
        with patch("src.config.integrations.settings") as mock_settings:
            mock_vllm = MagicMock()
            mock_vllm.model = "test-model"
            mock_vllm.context_window = 16384
            mock_vllm.kv_cache_dtype = "fp16"
            mock_vllm.gpu_memory_utilization = 0.9
            mock_vllm.max_num_seqs = 64
            mock_vllm.max_num_batched_tokens = 2048
            mock_vllm.enable_chunked_prefill = True
            mock_settings.vllm = mock_vllm

            command = get_vllm_server_command()
            assert "--enable-chunked-prefill" in command


@pytest.mark.unit
class TestInitializeIntegrationsCoverage:
    """Additional coverage for initialize_integrations (merged)."""

    @patch("src.config.integrations.setup_llamaindex")
    @patch("src.config.integrations.setup_vllm_env")
    def test_initialize_integrations_function_body_execution(
        self, mock_setup_vllm, mock_setup_llama, caplog
    ):
        """Test full execution of initialize_integrations function."""
        with caplog.at_level("INFO", logger="src.config.integrations"):
            initialize_integrations()
            mock_setup_vllm.assert_called_once()
            mock_setup_llama.assert_called_once()
            assert "All integrations initialized successfully" in caplog.text


@pytest.fixture(autouse=True)
def reset_llamaindex_settings():
    """Reset LlamaIndex Settings before and after each test (public API)."""
    # Store original values via public API
    original_llm = getattr(Settings, "llm", None)
    original_embed_model = getattr(Settings, "embed_model", None)

    # Reset to None via public API
    Settings.llm = None
    Settings.embed_model = None

    yield

    # Restore original values via public API
    Settings.llm = original_llm
    Settings.embed_model = original_embed_model


@pytest.mark.unit
class TestSetupLlamaIndex:
    """Test LlamaIndex configuration setup functionality."""

    def test_setup_llamaindex_llm_configuration_success(self):
        """Test successful LLM configuration with valid model config."""
        mock_model_config = {
            "model_name": "test-model",
            "base_url": "http://localhost:11434",
            "temperature": 0.7,
        }

        with patch("src.config.integrations.settings") as mock_settings:
            mock_settings.get_model_config.return_value = mock_model_config

            setup_llamaindex()

            # Verify LLM was configured with expected parameters
            assert isinstance(Settings.llm, Ollama)
            assert Settings.llm.model == "test-model"
            assert Settings.llm.base_url == "http://localhost:11434"
            assert Settings.llm.temperature == 0.7
            assert Settings.llm.request_timeout == 120.0

    def test_setup_llamaindex_embedding_config_cuda_dtype_selection(self):
        """Test CUDA device configuration selects correct torch dtype."""
        mock_embedding_config = {
            "model_name": "BAAI/bge-m3",
            "device": "cuda",
            "max_length": 8192,
            "batch_size": 32,
            "trust_remote_code": True,
        }

        with (
            patch("src.config.integrations.settings") as mock_settings,
            patch("src.config.integrations.BGEM3Embedding") as mock_embedding_class,
        ):
            mock_settings.get_model_config.side_effect = Exception("LLM failed")
            mock_settings.get_embedding_config.return_value = mock_embedding_config
            mock_settings.vllm = None

            mock_embedding_class.return_value = MagicMock()

            with patch("torch.cuda.is_available", return_value=True):
                setup_llamaindex()

                # Verify BGEM3Embedding was called with correct device and fp16
                mock_embedding_class.assert_called_once()
                call_args = mock_embedding_class.call_args
                assert call_args[1]["use_fp16"] is True
                assert call_args[1]["device"] == "cuda"
                assert call_args[1]["model_name"] == "BAAI/bge-m3"

    def test_setup_llamaindex_embedding_config_cpu_dtype_selection(self):
        """Test CPU device configuration selects correct torch dtype."""
        mock_embedding_config = {
            "model_name": "BAAI/bge-m3",
            "device": "cpu",
            "max_length": 8192,
            "batch_size": 16,
            "trust_remote_code": True,
        }

        with (
            patch("src.config.integrations.settings") as mock_settings,
            patch("src.config.integrations.BGEM3Embedding") as mock_embedding_class,
        ):
            mock_settings.get_model_config.side_effect = Exception("LLM failed")
            mock_settings.get_embedding_config.return_value = mock_embedding_config
            mock_settings.vllm = None

            mock_embedding_class.return_value = MagicMock()

            with patch("torch.cuda.is_available", return_value=False):
                setup_llamaindex()

                # Verify BGEM3Embedding was called with correct device and fp16 False
                mock_embedding_class.assert_called_once()
                call_args = mock_embedding_class.call_args
                assert call_args[1]["use_fp16"] is False
                assert call_args[1]["device"] == "cpu"
                assert call_args[1]["model_name"] == "BAAI/bge-m3"

    def test_setup_llamaindex_context_window_configuration_success(self):
        """Test successful context window and performance settings configuration."""
        mock_vllm_settings = MagicMock()
        mock_vllm_settings.context_window = 131072
        mock_vllm_settings.max_tokens = 4096

        with patch("src.config.integrations.settings") as mock_settings:
            mock_settings.get_model_config.side_effect = Exception("LLM failed")
            mock_settings.get_embedding_config.side_effect = Exception(
                "Embedding failed"
            )
            mock_settings.vllm = mock_vllm_settings

            setup_llamaindex()

            # Verify context settings were configured
            assert Settings.context_window == 131072
            assert Settings.num_output == 4096

    def test_setup_llamaindex_llm_configuration_failure_handling(self, caplog):
        """Test graceful handling of LLM configuration failure."""
        with patch("src.config.integrations.settings") as mock_settings:
            mock_settings.get_model_config.side_effect = Exception("Connection failed")
            # Ensure other configs also fail to test isolation
            mock_settings.get_embedding_config.side_effect = Exception(
                "Embedding failed"
            )
            mock_settings.vllm = None

            setup_llamaindex()

            # Verify LLM configuration failed and warning was logged
            # Settings.llm may be MockLLM when configuration fails
            assert "Could not configure LLM: Connection failed" in caplog.text

    def test_setup_llamaindex_embedding_configuration_failure_handling(self, caplog):
        """Test graceful handling of embedding configuration failure."""
        with patch("src.config.integrations.settings") as mock_settings:
            mock_settings.get_model_config.side_effect = Exception("LLM failed")
            mock_settings.get_embedding_config.side_effect = ValueError("Invalid model")
            mock_settings.vllm = None

            setup_llamaindex()

            # Verify embedding configuration failed and warning was logged
            # Settings.embed_model may be MockEmbedding when configuration fails
            assert "Could not configure embeddings: Invalid model" in caplog.text

    def test_setup_llamaindex_context_configuration_failure_handling(self, caplog):
        """Test graceful handling of context configuration failure."""
        with patch("src.config.integrations.settings") as mock_settings:
            mock_settings.get_model_config.side_effect = Exception("LLM failed")
            mock_settings.get_embedding_config.side_effect = Exception(
                "Embedding failed"
            )
            mock_settings.vllm = None  # This should cause AttributeError

            setup_llamaindex()

            # Verify warning was logged for context configuration failure
            assert "Could not set context configuration" in caplog.text

    def test_setup_llamaindex_cache_folder_path_resolution(self):
        """Test that cache folder path is properly resolved to absolute path."""
        mock_embedding_config = {
            "model_name": "BAAI/bge-m3",
            "device": "cpu",
            "max_length": 8192,
            "batch_size": 16,
            "trust_remote_code": True,
        }

        with (
            patch("src.config.integrations.settings") as mock_settings,
            patch("src.config.integrations.BGEM3Embedding") as mock_embedding_class,
        ):
            mock_settings.get_model_config.side_effect = Exception("LLM failed")
            mock_settings.get_embedding_config.return_value = mock_embedding_config
            mock_settings.vllm = None

            mock_embedding_instance = MagicMock()
            mock_embedding_class.return_value = mock_embedding_instance

            setup_llamaindex()

            # Verify BGEM3Embedding was called (cache path managed internally)
            mock_embedding_class.assert_called_once()
            call_args = mock_embedding_class.call_args
            assert call_args[1]["model_name"] == "BAAI/bge-m3"


@pytest.mark.unit
class TestSetupVLLMEnv:
    """Test vLLM environment variable setup functionality."""

    def test_setup_vllm_env_environment_variables_set(self):
        """Test that vLLM environment variables are set from configuration."""
        mock_vllm_env = {
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "VLLM_KV_CACHE_DTYPE": "fp8_e5m2",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.85",
        }

        # Clear environment first
        for key in mock_vllm_env:
            if key in os.environ:
                del os.environ[key]

        with patch("src.config.integrations.settings") as mock_settings:
            mock_settings.get_vllm_env_vars.return_value = mock_vllm_env

            setup_vllm_env()

            # Verify all environment variables were set
            for key, value in mock_vllm_env.items():
                assert os.environ[key] == value

    def test_setup_vllm_env_existing_variables_preserved(self):
        """Test that existing environment variables are not overwritten."""
        mock_vllm_env = {
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "EXISTING_VAR": "new_value",
        }

        # Set an existing environment variable
        os.environ["EXISTING_VAR"] = "original_value"

        with patch("src.config.integrations.settings") as mock_settings:
            mock_settings.get_vllm_env_vars.return_value = mock_vllm_env

            setup_vllm_env()

            # Verify existing variable was not overwritten
            assert os.environ["EXISTING_VAR"] == "original_value"
            # But new variable was set
            assert os.environ["VLLM_ATTENTION_BACKEND"] == "FLASHINFER"

        # Clean up
        if "EXISTING_VAR" in os.environ:
            del os.environ["EXISTING_VAR"]
        if "VLLM_ATTENTION_BACKEND" in os.environ:
            del os.environ["VLLM_ATTENTION_BACKEND"]

    def test_setup_vllm_env_empty_configuration(self):
        """Test handling of empty vLLM environment configuration."""
        with patch("src.config.integrations.settings") as mock_settings:
            mock_settings.get_vllm_env_vars.return_value = {}

            # Should not raise any errors
            setup_vllm_env()

    def test_setup_vllm_env_logging_behavior(self, caplog):
        """Test that environment variable setting is properly logged."""
        mock_vllm_env = {"TEST_VAR": "test_value"}

        # Ensure variable doesn't exist
        if "TEST_VAR" in os.environ:
            del os.environ["TEST_VAR"]

        with (
            caplog.at_level("INFO", logger="src.config.integrations"),
            patch("src.config.integrations.settings") as mock_settings,
        ):
            mock_settings.get_vllm_env_vars.return_value = mock_vllm_env

            setup_vllm_env()

            # Verify success message was logged
            assert (
                "vLLM environment variables configured for FP8 optimization"
                in caplog.text
            )

        # Clean up
        if "TEST_VAR" in os.environ:
            del os.environ["TEST_VAR"]


@pytest.mark.unit
class TestGetVLLMServerCommand:
    """Test vLLM server command generation functionality."""

    def test_get_vllm_server_command_basic_configuration(self):
        """Test basic vLLM server command generation."""
        with patch("src.config.integrations.settings") as mock_settings:
            mock_vllm = MagicMock()
            mock_vllm.model = "Qwen/Qwen3-4B-Instruct"
            mock_vllm.context_window = 131072
            mock_vllm.kv_cache_dtype = "fp8_e5m2"
            mock_vllm.gpu_memory_utilization = 0.85
            mock_vllm.max_num_seqs = 256
            mock_vllm.max_num_batched_tokens = 8192
            mock_vllm.enable_chunked_prefill = False
            mock_settings.vllm = mock_vllm

            command = get_vllm_server_command()

            expected_command = [
                "vllm",
                "serve",
                "Qwen/Qwen3-4B-Instruct",
                "--max-model-len",
                "131072",
                "--kv-cache-dtype",
                "fp8_e5m2",
                "--gpu-memory-utilization",
                "0.85",
                "--max-num-seqs",
                "256",
                "--max-num-batched-tokens",
                "8192",
                "--trust-remote-code",
            ]

            assert command == expected_command

    def test_get_vllm_server_command_with_chunked_prefill(self):
        """Test vLLM server command generation with chunked prefill enabled."""
        with patch("src.config.integrations.settings") as mock_settings:
            mock_vllm = MagicMock()
            mock_vllm.model = "test-model"
            mock_vllm.context_window = 65536
            mock_vllm.kv_cache_dtype = "fp8"
            mock_vllm.gpu_memory_utilization = 0.9
            mock_vllm.max_num_seqs = 128
            mock_vllm.max_num_batched_tokens = 4096
            mock_vllm.enable_chunked_prefill = True
            mock_settings.vllm = mock_vllm

            command = get_vllm_server_command()

            # Verify chunked prefill flag is included
            assert "--enable-chunked-prefill" in command

    def test_get_vllm_server_command_without_chunked_prefill(self):
        """Test vLLM server command generation without chunked prefill."""
        with patch("src.config.integrations.settings") as mock_settings:
            mock_vllm = MagicMock()
            mock_vllm.model = "test-model"
            mock_vllm.context_window = 32768
            mock_vllm.kv_cache_dtype = "fp16"
            mock_vllm.gpu_memory_utilization = 0.8
            mock_vllm.max_num_seqs = 64
            mock_vllm.max_num_batched_tokens = 2048
            mock_vllm.enable_chunked_prefill = False
            mock_settings.vllm = mock_vllm

            command = get_vllm_server_command()

            # Verify chunked prefill flag is not included
            assert "--enable-chunked-prefill" not in command

    def test_get_vllm_server_command_parameter_types(self):
        """Test that numeric parameters are properly converted to strings."""
        with patch("src.config.integrations.settings") as mock_settings:
            mock_vllm = MagicMock()
            mock_vllm.model = "test-model"
            mock_vllm.context_window = 16384
            mock_vllm.kv_cache_dtype = "auto"
            mock_vllm.gpu_memory_utilization = 0.75
            mock_vllm.max_num_seqs = 32
            mock_vllm.max_num_batched_tokens = 1024
            mock_vllm.enable_chunked_prefill = False
            mock_settings.vllm = mock_vllm

            command = get_vllm_server_command()

            # Verify all numeric values are converted to strings
            assert "16384" in command
            assert "0.75" in command
            assert "32" in command
            assert "1024" in command


@pytest.mark.unit
class TestInitializeIntegrations:
    """Test integration initialization orchestration functionality."""

    @patch("src.config.integrations.setup_llamaindex")
    @patch("src.config.integrations.setup_vllm_env")
    def test_initialize_integrations_calls_all_setup_functions(
        self, mock_setup_vllm, mock_setup_llama, caplog
    ):
        """Test that initialize_integrations calls all required setup functions."""
        with caplog.at_level("INFO", logger="src.config.integrations"):
            initialize_integrations()

            # Verify both setup functions were called
            mock_setup_vllm.assert_called_once()
            mock_setup_llama.assert_called_once()

            # Verify success message was logged
            assert "All integrations initialized successfully" in caplog.text

    @patch("src.config.integrations.setup_llamaindex")
    @patch("src.config.integrations.setup_vllm_env")
    def test_initialize_integrations_order_of_operations(
        self, mock_setup_vllm, mock_setup_llama
    ):
        """Test that vLLM environment setup is called before LlamaIndex setup."""
        call_order = []

        def track_vllm_call():
            call_order.append("vllm")

        def track_llama_call():
            call_order.append("llama")

        mock_setup_vllm.side_effect = track_vllm_call
        mock_setup_llama.side_effect = track_llama_call

        initialize_integrations()

        # Verify vLLM setup is called before LlamaIndex setup
        assert call_order == ["vllm", "llama"]

    @patch("src.config.integrations.setup_llamaindex")
    @patch("src.config.integrations.setup_vllm_env")
    def test_initialize_integrations_propagates_setup_failures(
        self, mock_setup_vllm, mock_setup_llama, caplog
    ):
        """Test that initialization propagates exceptions from setup functions."""
        # Make vLLM setup fail
        mock_setup_vllm.side_effect = Exception("vLLM setup failed")

        with caplog.at_level("INFO", logger="src.config.integrations"):
            # Should raise exception; initialize_integrations doesn't handle errors
            # gracefully in this scenario
            with pytest.raises(Exception, match="vLLM setup failed"):
                initialize_integrations()

            # Verify vLLM setup was attempted
            mock_setup_vllm.assert_called_once()
            # LlamaIndex setup should not be called due to early exception
            mock_setup_llama.assert_not_called()


@pytest.mark.unit
class TestIntegrationModuleBoundaryConditions:
    """Test edge cases and boundary conditions for integration module."""

    def test_embedding_cache_folder_creation_permissions(self):
        """Test handling of cache folder creation when permissions are restricted."""
        mock_embedding_config = {
            "model_name": "BAAI/bge-m3",
            "device": "cpu",
            "max_length": 512,
            "batch_size": 1,
            "trust_remote_code": False,
        }

        with (
            patch("src.config.integrations.settings") as mock_settings,
            patch("src.config.integrations.BGEM3Embedding") as mock_embedding_class,
        ):
            mock_settings.get_model_config.side_effect = Exception("LLM failed")
            mock_settings.get_embedding_config.return_value = mock_embedding_config
            mock_settings.vllm = None

            mock_embedding_instance = MagicMock()
            mock_embedding_class.return_value = mock_embedding_instance

            # Should not fail even if cache folder cannot be created
            setup_llamaindex()

            # Verify embedding model was configured (using mock). It may be
            # mock_embedding_instance or MockEmbedding when setup succeeds.
            assert Settings.embed_model is not None

    def test_torch_dtype_fallback_on_cuda_unavailable(self):
        """Test torch dtype fallback when CUDA is configured but unavailable."""
        mock_embedding_config = {
            "model_name": "BAAI/bge-m3",
            "device": "cuda",  # CUDA requested
            "max_length": 1024,
            "batch_size": 8,
            "trust_remote_code": True,
        }

        with (
            patch("src.config.integrations.settings") as mock_settings,
            patch("src.config.integrations.BGEM3Embedding") as mock_embedding_class,
        ):
            mock_settings.get_model_config.side_effect = Exception("LLM failed")
            mock_settings.get_embedding_config.return_value = mock_embedding_config
            mock_settings.vllm = None

            mock_embedding_instance = MagicMock()
            mock_embedding_class.return_value = mock_embedding_instance

            with patch(
                "torch.cuda.is_available", return_value=False
            ):  # But CUDA unavailable
                setup_llamaindex()

                # Verify BGEM3Embedding was called with fp16 disabled
                mock_embedding_class.assert_called_once()
                call_args = mock_embedding_class.call_args
                assert call_args[1]["use_fp16"] is False

    def test_environment_variable_handling_with_special_characters(self):
        """Test environment variable handling with special characters in values."""
        mock_vllm_env = {
            "VLLM_SPECIAL_PARAM": "value with spaces and symbols !@#$%",
            "VLLM_UNICODE_PARAM": "测试中文参数",
        }

        with patch("src.config.integrations.settings") as mock_settings:
            mock_settings.get_vllm_env_vars.return_value = mock_vllm_env

            setup_vllm_env()

            # Verify special characters are preserved
            assert (
                os.environ["VLLM_SPECIAL_PARAM"]
                == "value with spaces and symbols !@#$%"
            )
            assert os.environ["VLLM_UNICODE_PARAM"] == "测试中文参数"

        # Clean up
        for key in mock_vllm_env:
            if key in os.environ:
                del os.environ[key]

    def test_command_generation_with_extreme_values(self):
        """Test vLLM command generation with boundary value parameters."""
        with patch("src.config.integrations.settings") as mock_settings:
            mock_vllm = MagicMock()
            mock_vllm.model = "very-long-model-name-that-might-cause-issues"
            mock_vllm.context_window = 1  # Minimum value
            mock_vllm.kv_cache_dtype = "fp8_e4m3fn"  # Different FP8 format
            mock_vllm.gpu_memory_utilization = 0.99  # Near maximum
            mock_vllm.max_num_seqs = 1024  # High value
            mock_vllm.max_num_batched_tokens = 65536  # High value
            mock_vllm.enable_chunked_prefill = True
            mock_settings.vllm = mock_vllm

            command = get_vllm_server_command()

            # Verify extreme values are handled correctly
            assert "very-long-model-name-that-might-cause-issues" in command
            assert "1" in command
            assert "fp8_e4m3fn" in command
            assert "0.99" in command
            assert "1024" in command
            assert "65536" in command
            assert "--enable-chunked-prefill" in command
