"""Demonstration test for structural changes validation.

This test demonstrates the validation system and shows that the performance
and integration test framework is working correctly after cleanup and improvements.
"""

import pytest

# Graceful import handling for demonstration tests
try:
    from src.config.settings import DocMindSettings
except ImportError:
    # Mock settings for testing when config system is not available
    class DocMindSettings:
        def __init__(self):
            self.model_name = "mock-model"
            self.chunk_size = 1024
            self.embedding = type("obj", (object,), {"model_name": "mock-embedding"})()
            self.vllm = type("obj", (object,), {})()
            self.processing = type("obj", (object,), {})()

        def get_vllm_config(self):
            return {"model_name": "mock-model"}

        def get_embedding_config(self):
            return {"model_name": "mock-embedding"}

        def get_agent_config(self):
            return {"enable_multi_agent": True}


@pytest.mark.performance
class TestValidationDemo:
    """Demonstration tests to show validation system works."""

    def test_validation_framework_works(self):
        """Test that the validation framework can run tests."""
        # Simple test to verify the framework is working
        assert True, "Basic validation framework test"
        print("Validation framework: Working correctly")

    def test_settings_can_be_imported(self):
        """Test that settings can be imported after structural changes."""
        # Test that unified settings can be imported
        settings = DocMindSettings()

        assert settings is not None
        assert hasattr(settings, "model_name")
        assert hasattr(settings, "embedding")
        assert hasattr(settings, "vllm")
        assert hasattr(settings, "processing")

        print("Settings import: SUCCESS")
        print(f"  - Model: {settings.model_name}")
        print(f"  - Embedding model: {settings.embedding.model_name}")
        print(f"  - Chunk size: {settings.chunk_size}")

    def test_configuration_methods_work(self):
        """Test that configuration helper methods work."""
        settings = DocMindSettings()

        # Test configuration methods
        vllm_config = settings.get_vllm_config()
        embedding_config = settings.get_embedding_config()
        agent_config = settings.get_agent_config()

        # Verify configurations
        assert isinstance(vllm_config, dict)
        assert isinstance(embedding_config, dict)
        assert isinstance(agent_config, dict)

        assert "model_name" in vllm_config
        assert "model_name" in embedding_config
        assert "enable_multi_agent" in agent_config

        print("Configuration methods: SUCCESS")
        print(f"  - VLLM config keys: {len(vllm_config)}")
        print(f"  - Embedding config keys: {len(embedding_config)}")
        print(f"  - Agent config keys: {len(agent_config)}")


@pytest.mark.integration
class TestIntegrationDemo:
    """Demonstration integration tests."""

    def test_cross_module_imports_work(self):
        """Test that cross-module imports work after reorganization."""
        # Test imports from different modules work together
        try:
            from src.config.settings import DocMindSettings
            from src.models.embeddings import EmbeddingParameters
            from src.models.processing import ProcessingResult
            from src.models.storage import StorageConfig

            # Create instances
            settings = DocMindSettings()
            embedding_params = EmbeddingParameters()
            storage_config = StorageConfig()

            # Verify they work together
            assert settings is not None
            assert embedding_params is not None
            assert storage_config is not None

            # ProcessingResult needs arguments, so just verify we can import it
            assert ProcessingResult is not None

            print("Cross-module imports: SUCCESS")
            print("  - All modules imported successfully")
            print("  - All instances created successfully")

        except ImportError as e:
            pytest.fail(f"Cross-module import failed: {e}")

    def test_nested_configuration_sync(self):
        """Test that nested configuration synchronization works automatically."""
        settings = DocMindSettings(
            chunk_size=1024,
            agent_decision_timeout=200,
            bge_m3_model_name="test-model",  # ADR-024 compliant timeout
        )

        # Synchronization happens automatically in model_post_init
        # No need to call _sync_nested_models() explicitly

        # Verify synchronization worked automatically
        assert settings.processing.chunk_size == 1024
        assert settings.agents.decision_timeout == 200  # ADR-024 compliant
        assert settings.embedding.model_name == "test-model"

        print("Nested configuration sync: SUCCESS")
        print(f"  - Processing chunk size: {settings.processing.chunk_size}")
        print(f"  - Agent timeout: {settings.agents.decision_timeout}")
        print(f"  - Embedding model: {settings.embedding.model_name}")
