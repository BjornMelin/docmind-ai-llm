"""Comprehensive system validation for DocMind AI.

This module validates that all components can be imported and basic functionality
works correctly. It serves as a health check for the entire system integration
and verifies compatibility with the unified configuration architecture.

Focus Areas:
- Module import validation with current architecture
- Unified configuration system validation
- Core system health checks
- File structure validation for current components
"""

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.settings import DocMindSettings
from src.utils.core import detect_hardware, validate_startup_configuration


class TestImportValidation:
    """Validate all modules can be imported successfully."""

    def test_core_models_import(self):
        """Test that core models can be imported."""
        modules = [
            "src.models.schemas",
            "src.models.processing",
            "src.models.embeddings",
            "src.models.storage",
        ]

        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} imported but is None"

            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_src_utils_modules_import(self):
        """Test that src.utils modules can be imported."""
        modules = [
            "src.utils.core",
            "src.utils.monitoring",
            "src.utils.document",
            "src.utils.storage",
        ]

        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} imported but is None"

                # Test that key functions are available
                if module_name == "src.utils.core":
                    assert hasattr(module, "detect_hardware")
                    assert hasattr(module, "validate_startup_configuration")

            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_adr009_document_processing_modules_import(self):
        """Test that ADR-009 compliant document processing modules can be imported."""
        adr009_modules = [
            "src.processing.document_processor",
            "src.processing.chunking.unstructured_chunker",
            "src.cache.simple_cache",
            "src.processing.embeddings.bgem3_embedder",
        ]

        for module_name in adr009_modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} imported but is None"

                # Test that key classes are available
                if module_name == "src.processing.document_processor":
                    assert hasattr(module, "DocumentProcessor")
                elif module_name == "src.processing.chunking.unstructured_chunker":
                    assert hasattr(
                        module, "SemanticChunker"
                    )  # Actual class name in the module
                elif module_name == "src.cache.simple_cache":
                    assert hasattr(module, "SimpleCache")
                elif module_name == "src.processing.embeddings.bgem3_embedder":
                    assert hasattr(module, "BGEM3Embedder")

            except ImportError as e:
                pytest.fail(f"Failed to import ADR-009 module {module_name}: {e}")

    def test_agents_modules_import(self):
        """Test that agents modules can be imported."""
        modules = [
            "src.agents.coordinator",
            "src.agents.tool_factory",
            "src.agents.tools",
            "src.agents.retrieval",
        ]

        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} imported but is None"

            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    @pytest.mark.skipif(
        not importlib.util.find_spec("torch"), reason="PyTorch not available"
    )
    def test_hardware_detection_basic(self):
        """Test basic hardware detection functionality."""
        from src.utils.core import detect_hardware

        with patch("src.utils.core.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.cuda.device_count.return_value = 0

            hardware_info = detect_hardware()

            assert isinstance(hardware_info, dict)
            assert "cuda_available" in hardware_info
            assert hardware_info["cuda_available"] is False

    def test_basic_validation_integration(self):
        """Test that we can call basic validation functions."""
        settings = DocMindSettings()

        # validate_startup_configuration now requires a settings parameter
        try:
            result = validate_startup_configuration(settings)
            # Result should be a dict with validation results
            assert isinstance(result, dict)
            assert "valid" in result
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            pytest.fail(f"validate_startup_configuration failed: {e}")
        except Exception as e:
            # Handle network-related errors (e.g., Qdrant not running) gracefully
            # This is expected in unit test environments
            if "Connection refused" in str(e) or "qdrant" in str(e).lower():
                pytest.skip(f"Skipping test due to external dependency: {e}")
            else:
                pytest.fail(f"Unexpected error in validate_startup_configuration: {e}")

    def test_key_file_structure(self):
        """Test that essential files exist in correct locations."""
        base_path = Path(__file__).parent.parent.parent

        required_files = [
            "src/config/settings.py",
            "src/agents/tool_factory.py",
            "src/utils/core.py",
            "src/agents/coordinator.py",
        ]

        # ADR-009 compliant file structure
        adr009_files = [
            "src/processing/document_processor.py",
            "src/processing/chunking/unstructured_chunker.py",
            "src/cache/simple_cache.py",
            "src/processing/embeddings/bgem3_embedder.py",
        ]

        for file_path in required_files:
            full_path = base_path / file_path
            assert full_path.exists(), f"Required file missing: {full_path}"

        for file_path in adr009_files:
            full_path = base_path / file_path
            assert full_path.exists(), f"ADR-009 file missing: {full_path}"

    def test_basic_system_health(self):
        """Test basic system health check."""
        # Test that basic imports work
        try:
            # Create basic settings instance
            settings = DocMindSettings()
            assert settings is not None

            # Test hardware detection doesn't crash
            hardware_info = detect_hardware()
            assert isinstance(hardware_info, dict)

        except Exception as e:
            pytest.fail(f"Basic system health check failed: {e}")


class TestSettingsValidation:
    """Validate unified configuration settings."""

    def test_settings_creation(self):
        """Test that unified DocMindSettings can be created with defaults."""
        settings = DocMindSettings()
        assert settings is not None
        # Check for actual attributes that exist in the unified Settings class
        assert hasattr(settings, "qdrant_url")
        assert hasattr(settings, "enable_gpu_acceleration")
        assert hasattr(settings, "chunk_size")
        assert hasattr(settings, "model_name")
        assert hasattr(settings, "embedding_model")
        assert hasattr(settings, "llm_backend")

    def test_settings_required_fields(self):
        """Test that unified settings have all required fields."""
        settings = DocMindSettings()
        # These should exist and have reasonable defaults
        # Note: chunk_size moved to nested processing config, chunk_overlap was removed
        assert hasattr(settings.processing, "chunk_size")
        assert isinstance(settings.processing.chunk_size, int)

        # Test additional key settings that should exist
        # (now in nested embedding config)
        assert hasattr(settings.embedding, "dimension")
        assert hasattr(settings.embedding, "model_name")
        assert hasattr(settings, "top_k")

        # Test nested config access
        assert hasattr(settings, "vllm")
        assert hasattr(settings, "agents")
        assert hasattr(settings, "retrieval")
        assert hasattr(settings, "embedding")

    def test_unified_config_structure(self):
        """Test that unified configuration structure is properly setup."""
        settings = DocMindSettings()

        # Test nested configurations are properly initialized
        assert settings.vllm is not None
        assert settings.agents is not None
        assert settings.retrieval is not None
        assert settings.embedding is not None

        # Test that nested configs have expected attributes
        assert hasattr(settings.vllm, "model")
        assert hasattr(settings.agents, "enable_multi_agent")
        assert hasattr(settings.retrieval, "top_k")
        assert hasattr(settings.embedding, "model_name")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
