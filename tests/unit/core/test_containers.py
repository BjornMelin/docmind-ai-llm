"""Tests for dependency injection container (cache-less architecture).

Validates provider wiring for embedding model, document processor, and
multi-agent coordinator, and environment-driven configuration loading.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from dependency_injector import providers

from src.containers import (
    ApplicationContainer,
    create_container,
    create_document_processor,
    create_embedding_model,
    create_multi_agent_coordinator,
    get_container,
)


@pytest.mark.unit
class TestApplicationContainer:
    """Test ApplicationContainer configuration and provider setup."""

    def test_application_container_provider_configuration(self):
        """Test that ApplicationContainer has all expected providers configured."""
        container = ApplicationContainer()

        # Verify required providers are present (cache removed per ADR-030)
        assert hasattr(container, "config")
        assert hasattr(container, "embedding_model")
        assert hasattr(container, "document_processor")
        assert hasattr(container, "multi_agent_coordinator")

        # Verify provider types
        assert isinstance(container.config, providers.Configuration)
        assert isinstance(container.embedding_model, providers.Factory)
        assert isinstance(container.document_processor, providers.Factory)
        assert isinstance(container.multi_agent_coordinator, providers.Singleton)

    def test_application_container_embedding_model_provider_configuration(self):
        """Test embedding model provider configuration with defaults."""
        container = ApplicationContainer()

        # Configure with default embedding settings
        container.config.from_dict(
            {
                "embedding_model": "BAAI/bge-m3",
                "use_fp16": True,
                "device": "cuda",
            }
        )

        # Verify embedding model provider configuration
        embedding_provider = container.embedding_model
        provides = embedding_provider.provides
        if hasattr(provides, "__name__"):
            assert provides.__name__ == "create_bgem3_embedding"
        else:
            provides_str = str(provides)
            assert "create_bgem3_embedding" in provides_str

    def test_application_container_document_processor_provider_configuration(self):
        """Test document processor provider configuration."""
        container = ApplicationContainer()
        processor_provider = container.document_processor
        provides_str = str(processor_provider.provides)
        assert "src.processing.document_processor.DocumentProcessor" in provides_str

    def test_application_container_multi_agent_coordinator_provider_configuration(self):
        """Test multi-agent coordinator provider configuration with defaults."""
        container = ApplicationContainer()
        container.config.from_dict(
            {
                "model_path": "Qwen/Qwen3-4B-Instruct",
                "max_context_length": 131072,
                "enable_fallback": True,
            }
        )
        coordinator_provider = container.multi_agent_coordinator
        assert isinstance(coordinator_provider, providers.Singleton)
        provides_str = str(coordinator_provider.provides)
        assert "src.agents.coordinator.MultiAgentCoordinator" in provides_str


@pytest.mark.unit
class TestContainerBasics:
    """Basic container behavior (single ApplicationContainer implementation)."""

    def test_get_container_returns_application(self):
        """Container exposes expected providers in runtime container."""
        c = get_container()
        assert hasattr(c, "config")
        assert hasattr(c, "embedding_model")
        assert hasattr(c, "document_processor")
        assert hasattr(c, "multi_agent_coordinator")

    @patch.dict(os.environ, {}, clear=True)
    def test_create_container_production_environment(self, caplog):
        """Test container creation in default environment."""
        if "TESTING" in os.environ:
            del os.environ["TESTING"]
        if "PYTEST_CURRENT_TEST" in os.environ:
            del os.environ["PYTEST_CURRENT_TEST"]

        container = create_container()
        # DI library returns a DynamicContainer instance; verify expected providers
        assert hasattr(container, "config")
        assert "Created ApplicationContainer" in caplog.text

    @patch.dict(os.environ, {"TESTING": "1"}, clear=True)
    def test_create_container_testing_environment_with_testing_flag(self, caplog):
        """Test container creation with TESTING environment variable."""
        container = create_container()
        assert hasattr(container, "config")
        assert "Created ApplicationContainer" in caplog.text

    @patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "test_example.py"}, clear=True)
    def test_create_container_testing_environment_with_pytest_flag(self, caplog):
        """Test container creation with PYTEST_CURRENT_TEST environment variable."""
        container = create_container()
        assert hasattr(container, "config")
        assert "Created ApplicationContainer" in caplog.text

    @patch.dict(os.environ, {"TESTING": "0", "PYTEST_CURRENT_TEST": ""}, clear=True)
    def test_create_container_production_with_false_testing_flags(self, caplog):
        """Test container creation with false-y testing flags."""
        container = create_container()
        assert hasattr(container, "config")
        assert "Created ApplicationContainer" in caplog.text


@pytest.mark.unit
class MockContainerConfiguration:
    """Test container configuration loading from environment variables."""

    @patch.dict(
        os.environ,
        {
            "DOCMIND_CACHE_DIR": "/custom/cache",
            "DOCMIND_EMBEDDING_MODEL": "custom-model",
            "DOCMIND_MODEL_NAME": "custom-llm-model",
            "DOCMIND_CONTEXT_WINDOW_SIZE": "65536",
            "DOCMIND_DEVICE": "cpu",
            "DOCMIND_USE_FP16": "false",
            "DOCMIND_ENABLE_FALLBACK_RAG": "false",
        },
        clear=True,
    )
    @patch("src.containers.settings")
    def test_create_container_environment_variable_loading(self, mock_settings):
        """Test that environment variables are properly loaded into configuration."""
        mock_settings.embedding.model_name = "default-embedding"
        mock_settings.vllm.model = "default-llm"
        mock_settings.vllm.context_window = 131072

        container = create_container()
        config = container.config()
        assert config["cache_dir"] == "/custom/cache"
        assert config["embedding_model"] == "custom-model"
        assert config["model_path"] == "custom-llm-model"
        assert config["max_context_length"] == 65536
        assert config["device"] == "cpu"
        assert config["use_fp16"] is False
        assert config["enable_fallback"] is False

    @patch.dict(os.environ, {}, clear=True)
    @patch("src.containers.settings")
    def test_create_container_default_configuration_fallback(self, mock_settings):
        """Test that default configuration is used when env vars are not set."""
        mock_settings.embedding.model_name = "BAAI/bge-m3"
        mock_settings.vllm.model = "Qwen/Qwen3-4B-Instruct"
        mock_settings.vllm.context_window = 131072

        container = create_container()
        config = container.config()
        assert config["cache_dir"] == "./cache"
        assert config["embedding_model"] == "BAAI/bge-m3"
        assert config["model_path"] == "Qwen/Qwen3-4B-Instruct"
        assert config["max_context_length"] == 131072
        assert config["device"] == "cuda"
        assert config["use_fp16"] is True
        assert config["enable_fallback"] is True

    @patch.dict(os.environ, {"DOCMIND_CONTEXT_WINDOW_SIZE": "invalid"}, clear=True)
    @patch("src.containers.settings")
    def test_create_container_invalid_numeric_environment_variable(self, mock_settings):
        """Test handling of invalid numeric environment variables."""
        mock_settings.vllm.context_window = 131072

        with pytest.raises(ValueError, match="invalid|int|convert|context|window"):
            create_container()

    @patch.dict(os.environ, {"DOCMIND_USE_FP16": "invalid_boolean"}, clear=True)
    @patch("src.containers.settings")
    def test_create_container_invalid_boolean_environment_variable(self, mock_settings):
        """Test handling of invalid boolean environment variables."""
        mock_settings.embedding.model_name = "test-model"
        mock_settings.vllm.model = "test-llm"
        mock_settings.vllm.context_window = 131072

        container = create_container()
        config = container.config()
        assert config["use_fp16"] is False


@pytest.mark.unit
class TestFactoryFunctions:
    """Test factory functions for manual instantiation (no cache)."""

    @patch("src.containers.container")
    def test_create_embedding_model_factory_function(self, mock_container):
        """Factory returns embedding model instance."""
        mock_embedding_instance = MagicMock()
        mock_container.embedding_model.return_value = mock_embedding_instance

        result = create_embedding_model(model_name="custom-model")

        mock_container.embedding_model.assert_called_once_with(
            model_name="custom-model"
        )
        assert result is mock_embedding_instance

    @patch("src.containers.container")
    def test_create_document_processor_factory_function(self, mock_container):
        """Factory returns document processor instance."""
        mock_processor_instance = MagicMock()
        mock_container.document_processor.return_value = mock_processor_instance

        result = create_document_processor(custom_param="value")

        mock_container.document_processor.assert_called_once_with(custom_param="value")
        assert result is mock_processor_instance

    @patch("src.containers.container")
    def test_create_multi_agent_coordinator_factory_function(self, mock_container):
        """Factory returns multi-agent coordinator instance."""
        mock_coordinator_instance = MagicMock()
        mock_container.multi_agent_coordinator.return_value = mock_coordinator_instance

        result = create_multi_agent_coordinator(model_path="custom-model")

        mock_container.multi_agent_coordinator.assert_called_once_with(
            model_path="custom-model"
        )
        assert result is mock_coordinator_instance

    @patch("src.containers.container")
    def test_factory_functions_with_no_arguments(self, mock_container):
        """Factories can be called without arguments."""
        mock_container.embedding_model.return_value = MagicMock()
        mock_container.document_processor.return_value = MagicMock()
        mock_container.multi_agent_coordinator.return_value = MagicMock()

        create_embedding_model()
        create_document_processor()
        create_multi_agent_coordinator()

        mock_container.embedding_model.assert_called_once_with()
        mock_container.document_processor.assert_called_once_with()
        mock_container.multi_agent_coordinator.assert_called_once_with()


@pytest.mark.unit
class MockContainerSingletonBehavior:
    """Test singleton provider behavior in containers (no cache)."""

    def test_singleton_coordinator_provider_reuse(self):
        """Coordinator provider returns the same instance (singleton)."""
        container = ApplicationContainer()
        container.config.from_dict(
            {
                "model_path": "test-model",
                "max_context_length": 4096,
                "enable_fallback": True,
            }
        )

        with patch("src.agents.coordinator.MultiAgentCoordinator") as mock_coord:
            mock_instance = MagicMock()
            mock_coord.return_value = mock_instance

            coord1 = container.multi_agent_coordinator()
            coord2 = container.multi_agent_coordinator()

            assert coord1 is coord2
            mock_coord.assert_called_once_with(
                model_path="test-model",
                max_context_length=4096,
                enable_fallback=True,
            )
