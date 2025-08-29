"""Comprehensive tests for dependency injection container.

Tests for ApplicationContainer, TestContainer, and dependency injection
functionality including configuration loading, provider instantiation,
singleton behavior, and environment-based container selection.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from dependency_injector import providers

from src.containers import (
    ApplicationContainer,
    TestContainer,
    create_cache,
    create_container,
    create_document_processor,
    create_embedding_model,
    create_multi_agent_coordinator,
    get_cache,
    get_container,
    get_document_processor,
    get_embedding_model,
    get_multi_agent_coordinator,
    unwire_container,
    wire_container,
)


@pytest.mark.unit
class TestApplicationContainer:
    """Test ApplicationContainer configuration and provider setup."""

    def test_application_container_provider_configuration(self):
        """Test that ApplicationContainer has all expected providers configured."""
        container = ApplicationContainer()

        # Verify all required providers are present
        assert hasattr(container, "config")
        assert hasattr(container, "cache")
        assert hasattr(container, "embedding_model")
        assert hasattr(container, "document_processor")
        assert hasattr(container, "multi_agent_coordinator")

        # Verify provider types
        assert isinstance(container.config, providers.Configuration)
        assert isinstance(container.cache, providers.Singleton)
        assert isinstance(container.embedding_model, providers.Factory)
        assert isinstance(container.document_processor, providers.Factory)
        assert isinstance(container.multi_agent_coordinator, providers.Singleton)

    def test_application_container_cache_provider_configuration(self):
        """Test cache provider configuration with default cache directory."""
        container = ApplicationContainer()

        # Configure with default cache directory
        container.config.from_dict({"cache_dir": "./cache"})

        # Verify cache provider is configured - use string representation of the provides attribute
        cache_provider = container.cache
        provides_str = str(cache_provider.provides)
        assert "src.cache.simple_cache.SimpleCache" in provides_str

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
        # For function providers, check if the provider provides the right function
        provides = embedding_provider.provides
        if hasattr(provides, "__name__"):
            assert provides.__name__ == "create_bgem3_embedding"
        else:
            provides_str = str(provides)
            assert "create_bgem3_embedding" in provides_str

    def test_application_container_document_processor_provider_configuration(self):
        """Test document processor provider configuration."""
        container = ApplicationContainer()

        # Verify document processor provider configuration
        processor_provider = container.document_processor
        provides_str = str(processor_provider.provides)
        assert "src.processing.document_processor.DocumentProcessor" in provides_str

    def test_application_container_multi_agent_coordinator_provider_configuration(self):
        """Test multi-agent coordinator provider configuration with defaults."""
        container = ApplicationContainer()

        # Configure with default agent settings
        container.config.from_dict(
            {
                "model_path": "Qwen/Qwen3-4B-Instruct",
                "max_context_length": 131072,
                "enable_fallback": True,
            }
        )

        # Verify multi-agent coordinator provider is singleton
        coordinator_provider = container.multi_agent_coordinator
        assert isinstance(coordinator_provider, providers.Singleton)
        provides_str = str(coordinator_provider.provides)
        assert "src.agents.coordinator.MultiAgentCoordinator" in provides_str


@pytest.mark.unit
class TestTestContainer:
    """Test TestContainer configuration and mock provider overrides."""

    def test_test_container_inherits_from_application_container(self):
        """Test that TestContainer inherits from ApplicationContainer."""
        assert issubclass(TestContainer, ApplicationContainer)

    def test_test_container_provider_overrides(self):
        """Test that TestContainer overrides providers with mocks."""
        container = TestContainer()

        # Verify mock providers are configured
        cache_provides_str = str(container.cache.provides)
        embedding_provides_str = str(container.embedding_model.provides)
        processor_provides_str = str(container.document_processor.provides)

        assert "tests._mocks.cache.MockCache" in cache_provides_str
        assert "tests._mocks.embeddings.MockEmbeddingModel" in embedding_provides_str
        assert "tests._mocks.processing.MockDocumentProcessor" in processor_provides_str

        # Verify provider types are maintained
        assert isinstance(
            container.cache, providers.Factory
        )  # Changed from Singleton to Factory
        assert isinstance(container.embedding_model, providers.Factory)
        assert isinstance(container.document_processor, providers.Factory)

    def test_test_container_retains_base_providers(self):
        """Test that TestContainer retains non-overridden providers from base."""
        container = TestContainer()

        # Verify non-overridden providers are inherited
        assert hasattr(container, "config")
        assert hasattr(container, "multi_agent_coordinator")
        assert isinstance(container.config, providers.Configuration)
        assert isinstance(container.multi_agent_coordinator, providers.Singleton)


@pytest.mark.unit
class TestContainerCreation:
    """Test container creation logic and environment-based selection."""

    @patch.dict(os.environ, {}, clear=True)
    def test_create_container_production_environment(self, caplog):
        """Test container creation in production environment."""
        # Ensure testing environment variables are not set
        if "TESTING" in os.environ:
            del os.environ["TESTING"]
        if "PYTEST_CURRENT_TEST" in os.environ:
            del os.environ["PYTEST_CURRENT_TEST"]

        container = create_container()

        # Verify ApplicationContainer is created
        assert isinstance(container, ApplicationContainer)
        assert not isinstance(container, TestContainer)
        assert "Created ApplicationContainer for production environment" in caplog.text

    @patch.dict(os.environ, {"TESTING": "1"}, clear=True)
    def test_create_container_testing_environment_with_testing_flag(self, caplog):
        """Test container creation with TESTING environment variable."""
        container = create_container()

        # Verify TestContainer is created
        assert isinstance(container, TestContainer)
        assert "Created TestContainer for testing environment" in caplog.text

    @patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "test_example.py"}, clear=True)
    def test_create_container_testing_environment_with_pytest_flag(self, caplog):
        """Test container creation with PYTEST_CURRENT_TEST environment variable."""
        container = create_container()

        # Verify TestContainer is created
        assert isinstance(container, TestContainer)
        assert "Created TestContainer for testing environment" in caplog.text

    @patch.dict(os.environ, {"TESTING": "0", "PYTEST_CURRENT_TEST": ""}, clear=True)
    def test_create_container_production_with_false_testing_flags(self, caplog):
        """Test container creation with false-y testing flags."""
        container = create_container()

        # Verify ApplicationContainer is created (false-y values don't trigger test mode)
        assert isinstance(container, ApplicationContainer)
        assert not isinstance(container, TestContainer)
        assert "Created ApplicationContainer for production environment" in caplog.text


@pytest.mark.unit
class TestContainerConfiguration:
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
        # Mock settings defaults
        mock_settings.embedding.model_name = "default-embedding"
        mock_settings.vllm.model = "default-llm"
        mock_settings.vllm.context_window = 131072

        container = create_container()

        # Verify environment variables were loaded
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
        """Test that default configuration is used when environment variables are not set."""
        # Mock settings defaults
        mock_settings.embedding.model_name = "BAAI/bge-m3"
        mock_settings.vllm.model = "Qwen/Qwen3-4B-Instruct"
        mock_settings.vllm.context_window = 131072

        container = create_container()

        # Verify defaults were used
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

        # Should raise ValueError for invalid integer conversion
        with pytest.raises(ValueError):
            create_container()

    @patch.dict(os.environ, {"DOCMIND_USE_FP16": "invalid_boolean"}, clear=True)
    @patch("src.containers.settings")
    def test_create_container_invalid_boolean_environment_variable(self, mock_settings):
        """Test handling of invalid boolean environment variables."""
        mock_settings.embedding.model_name = "test-model"
        mock_settings.vllm.model = "test-llm"
        mock_settings.vllm.context_window = 131072

        container = create_container()

        # Invalid boolean should default to False
        config = container.config()
        assert config["use_fp16"] is False


@pytest.mark.unit
class TestContainerWiring:
    """Test container wiring and unwiring functionality."""

    def test_wire_container_with_modules(self):
        """Test wiring container with specified modules."""
        # Create a fresh container for testing
        test_container = ApplicationContainer()

        # Mock the wire method
        with patch.object(test_container, "wire") as mock_wire:
            # Temporarily replace global container
            with patch("src.containers.container", test_container):
                wire_container(["src.app", "src.agents"])
                mock_wire.assert_called_once_with(modules=["src.app", "src.agents"])

    def test_unwire_container(self):
        """Test unwiring container functionality."""
        # Create a fresh container for testing
        test_container = ApplicationContainer()

        # Mock the unwire method
        with patch.object(test_container, "unwire") as mock_unwire:
            # Temporarily replace global container
            with patch("src.containers.container", test_container):
                unwire_container()
                mock_unwire.assert_called_once()

    def test_get_container_returns_global_instance(self):
        """Test that get_container returns the global container instance."""
        container_instance = get_container()

        # Should return the global container (either Application or Test based on environment)
        assert isinstance(container_instance, ApplicationContainer | TestContainer)


@pytest.mark.unit
class TestDependencyInjectionFunctions:
    """Test dependency injection functions with provider overrides."""

    def test_get_cache_dependency_injection(self):
        """Test cache dependency injection function."""
        mock_cache = MagicMock()

        # Test with manual injection
        result = get_cache(cache=mock_cache)
        assert result is mock_cache

    def test_get_embedding_model_dependency_injection(self):
        """Test embedding model dependency injection function."""
        mock_embedding = MagicMock()

        # Test with manual injection
        result = get_embedding_model(embedding_model=mock_embedding)
        assert result is mock_embedding

    def test_get_document_processor_dependency_injection(self):
        """Test document processor dependency injection function."""
        mock_processor = MagicMock()

        # Test with manual injection
        result = get_document_processor(processor=mock_processor)
        assert result is mock_processor

    def test_get_multi_agent_coordinator_dependency_injection(self):
        """Test multi-agent coordinator dependency injection function."""
        mock_coordinator = MagicMock()

        # Test with manual injection
        result = get_multi_agent_coordinator(coordinator=mock_coordinator)
        assert result is mock_coordinator


@pytest.mark.unit
class TestFactoryFunctions:
    """Test factory functions for manual instantiation."""

    @patch("src.containers.container")
    def test_create_cache_factory_function(self, mock_container):
        """Test cache factory function."""
        mock_cache_instance = MagicMock()
        mock_container.cache.return_value = mock_cache_instance

        result = create_cache(cache_dir="/test/cache")

        mock_container.cache.assert_called_once_with(cache_dir="/test/cache")
        assert result is mock_cache_instance

    @patch("src.containers.container")
    def test_create_embedding_model_factory_function(self, mock_container):
        """Test embedding model factory function."""
        mock_embedding_instance = MagicMock()
        mock_container.embedding_model.return_value = mock_embedding_instance

        result = create_embedding_model(model_name="custom-model")

        mock_container.embedding_model.assert_called_once_with(
            model_name="custom-model"
        )
        assert result is mock_embedding_instance

    @patch("src.containers.container")
    def test_create_document_processor_factory_function(self, mock_container):
        """Test document processor factory function."""
        mock_processor_instance = MagicMock()
        mock_container.document_processor.return_value = mock_processor_instance

        result = create_document_processor(custom_param="value")

        mock_container.document_processor.assert_called_once_with(custom_param="value")
        assert result is mock_processor_instance

    @patch("src.containers.container")
    def test_create_multi_agent_coordinator_factory_function(self, mock_container):
        """Test multi-agent coordinator factory function."""
        mock_coordinator_instance = MagicMock()
        mock_container.multi_agent_coordinator.return_value = mock_coordinator_instance

        result = create_multi_agent_coordinator(model_path="custom-model")

        mock_container.multi_agent_coordinator.assert_called_once_with(
            model_path="custom-model"
        )
        assert result is mock_coordinator_instance

    @patch("src.containers.container")
    def test_factory_functions_with_no_arguments(self, mock_container):
        """Test factory functions called without arguments."""
        # Mock return values
        mock_container.cache.return_value = MagicMock()
        mock_container.embedding_model.return_value = MagicMock()
        mock_container.document_processor.return_value = MagicMock()
        mock_container.multi_agent_coordinator.return_value = MagicMock()

        # Call factory functions without arguments
        create_cache()
        create_embedding_model()
        create_document_processor()
        create_multi_agent_coordinator()

        # Verify all were called with no arguments
        mock_container.cache.assert_called_once_with()
        mock_container.embedding_model.assert_called_once_with()
        mock_container.document_processor.assert_called_once_with()
        mock_container.multi_agent_coordinator.assert_called_once_with()


@pytest.mark.unit
class TestContainerSingletonBehavior:
    """Test singleton provider behavior in containers."""

    def test_singleton_cache_provider_reuse(self):
        """Test that cache singleton provider returns same instance."""
        container = ApplicationContainer()
        container.config.from_dict({"cache_dir": "./test_cache"})

        # Mock the cache class to track instantiation
        with patch("src.cache.simple_cache.SimpleCache") as MockCache:
            mock_instance = MagicMock()
            MockCache.return_value = mock_instance

            # Get cache instance twice
            cache1 = container.cache()
            cache2 = container.cache()

            # Should be the same instance (singleton behavior)
            assert cache1 is cache2
            # Constructor should only be called once
            MockCache.assert_called_once_with(cache_dir="./test_cache")

    def test_singleton_coordinator_provider_reuse(self):
        """Test that coordinator singleton provider returns same instance."""
        container = ApplicationContainer()
        container.config.from_dict(
            {
                "model_path": "test-model",
                "max_context_length": 4096,
                "enable_fallback": True,
            }
        )

        # Mock the coordinator class to track instantiation
        with patch("src.agents.coordinator.MultiAgentCoordinator") as MockCoordinator:
            mock_instance = MagicMock()
            MockCoordinator.return_value = mock_instance

            # Get coordinator instance twice
            coord1 = container.multi_agent_coordinator()
            coord2 = container.multi_agent_coordinator()

            # Should be the same instance (singleton behavior)
            assert coord1 is coord2
            # Constructor should only be called once
            MockCoordinator.assert_called_once_with(
                model_path="test-model",
                max_context_length=4096,
                enable_fallback=True,
            )

    def test_factory_provider_creates_new_instances(self):
        """Test that factory providers create new instances each time."""
        container = ApplicationContainer()
        container.config.from_dict(
            {
                "embedding_model": "test-model",
                "use_fp16": False,
                "device": "cpu",
            }
        )

        # Mock the embedding factory to track instantiation
        with patch("src.retrieval.embeddings.create_bgem3_embedding") as MockEmbedding:
            mock_instance1 = MagicMock()
            mock_instance2 = MagicMock()
            MockEmbedding.side_effect = [mock_instance1, mock_instance2]

            # Get embedding instance twice
            embed1 = container.embedding_model()
            embed2 = container.embedding_model()

            # Should be different instances (factory behavior)
            assert embed1 is not embed2
            assert embed1 is mock_instance1
            assert embed2 is mock_instance2
            # Constructor should be called twice
            assert MockEmbedding.call_count == 2


@pytest.mark.unit
class TestContainerEdgeCases:
    """Test edge cases and error conditions for container functionality."""

    @patch.dict(os.environ, {"TESTING": "1", "PYTEST_CURRENT_TEST": "test"}, clear=True)
    def test_create_container_both_testing_flags_set(self, caplog):
        """Test container creation when both testing flags are set."""
        container = create_container()

        # Should still create TestContainer (either flag is sufficient)
        assert isinstance(container, TestContainer)
        assert "Created TestContainer for testing environment" in caplog.text

    def test_container_configuration_with_empty_environment_prefix(self):
        """Test container configuration loading with empty DOCMIND prefix."""
        container = ApplicationContainer()

        # Should not raise errors when loading empty configuration
        container.config.from_env("DOCMIND")

        # Configuration should be accessible even if empty
        config = container.config()
        assert isinstance(config, dict)

    @patch("src.containers.settings", None)
    def test_create_container_with_missing_settings_import(self):
        """Test container creation when settings module is unavailable."""
        # Should raise AttributeError when trying to access settings attributes
        with pytest.raises(AttributeError):
            create_container()

    def test_container_provider_override_inheritance(self):
        """Test that TestContainer properly overrides parent providers."""
        app_container = ApplicationContainer()
        test_container = TestContainer()

        # Verify that overridden providers are different
        assert str(app_container.cache.provides) != str(test_container.cache.provides)
        assert str(app_container.embedding_model.provides) != str(
            test_container.embedding_model.provides
        )
        assert str(app_container.document_processor.provides) != str(
            test_container.document_processor.provides
        )

        # Verify that non-overridden providers are the same
        assert str(app_container.config.provides) == str(test_container.config.provides)
        assert str(app_container.multi_agent_coordinator.provides) == str(
            test_container.multi_agent_coordinator.provides
        )

    def test_container_configuration_type_coercion_edge_cases(self):
        """Test configuration type coercion with edge case values."""
        container = ApplicationContainer()

        # Test various boolean representations
        test_configs = [
            ({"use_fp16": "TRUE"}, True),  # Uppercase
            ({"use_fp16": "True"}, True),  # Capitalized
            ({"use_fp16": "1"}, False),  # Numeric string (not 'true')
            ({"use_fp16": "yes"}, False),  # Non-standard boolean
            ({"use_fp16": ""}, False),  # Empty string
        ]

        for config_dict, expected in test_configs:
            container.config.from_dict(config_dict)
            config = container.config()
            assert config.get("use_fp16") == expected
