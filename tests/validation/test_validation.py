"""Comprehensive test validation module for DocMind AI.

This module validates that all components can be imported and basic functionality
works correctly. It serves as a health check for the entire test suite and
system integration.

Note: Many utils modules were removed during cleanup. Only testing existing modules.

Following PyTestQA-Agent standards for comprehensive testing.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import importlib
from unittest.mock import patch

import pytest


class TestImportValidation:
    """Validate all modules can be imported successfully."""

    def test_core_models_import(self):
        """Test that core models can be imported."""
        modules = [
            "src.models.core",
        ]

        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} imported but is None"

                # Test that key classes are available
                if module_name == "src.models.core":
                    assert hasattr(module, "AppSettings")
                    assert hasattr(module, "AnalysisOutput")
                    assert hasattr(module, "Settings")
                    assert hasattr(module, "settings")

            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_src_utils_modules_import(self):
        """Test that src.utils modules can be imported."""
        modules = [
            "src.utils.core",
            "src.utils.document",
            "src.utils.embedding",
            "src.utils.database",
            "src.utils.monitoring",
        ]

        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} imported but is None"

                # Test that key functions are available
                if module_name == "src.utils.core":
                    assert hasattr(module, "detect_hardware")
                    assert hasattr(module, "validate_startup_configuration")
                elif module_name == "src.utils.document":
                    assert hasattr(module, "load_documents_unstructured")
                    assert hasattr(module, "ensure_spacy_model")
                elif module_name == "src.utils.embedding":
                    assert hasattr(module, "create_vector_index_async")
                    assert hasattr(module, "get_embed_model")

            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_agents_modules_import(self):
        """Test that agents modules can be imported."""
        modules = [
            "src.agents.agent_utils",
        ]

        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} imported but is None"

                # Test that key functions are available
                if module_name == "src.agents.agent_utils":
                    assert hasattr(module, "create_tools_from_index")

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
        from src.models.core import AppSettings
        from src.utils.core import validate_startup_configuration

        # validate_startup_configuration now requires a settings parameter
        try:
            settings = AppSettings()
            result = validate_startup_configuration(settings)
            # Result should be a dict with validation results
            assert isinstance(result, dict)
            assert "valid" in result
        except Exception as e:
            pytest.fail(f"validate_startup_configuration failed: {e}")

    def test_spacy_model_function_exists(self):
        """Test that spaCy model function exists and is callable."""
        from src.utils.document import ensure_spacy_model

        assert callable(ensure_spacy_model)

    def test_key_file_structure(self):
        """Test that essential files exist in correct locations."""
        base_path = Path(__file__).parent.parent.parent

        required_files = [
            "src/models/core.py",
            "src/agents/agent_factory.py",
            "src/utils/core.py",
            "src/utils/document.py",
            "src/utils/embedding.py",
        ]

        for file_path in required_files:
            full_path = base_path / file_path
            assert full_path.exists(), f"Required file missing: {full_path}"

    def test_basic_system_health(self):
        """Test basic system health check."""
        # Test that basic imports work
        try:
            from src.models.core import AppSettings
            from src.utils.core import detect_hardware

            # Create basic settings instance
            settings = AppSettings()
            assert settings is not None

            # Test hardware detection doesn't crash
            hardware_info = detect_hardware()
            assert isinstance(hardware_info, dict)

        except Exception as e:
            pytest.fail(f"Basic system health check failed: {e}")


class TestSettingsValidation:
    """Validate settings and configuration."""

    def test_settings_creation(self):
        """Test that settings can be created with defaults."""
        from src.models.core import AppSettings

        settings = AppSettings()
        assert settings is not None
        # Check for actual attributes that exist in the simplified Settings class
        assert hasattr(settings, "qdrant_url")
        assert hasattr(settings, "gpu_acceleration")
        assert hasattr(settings, "chunk_size")

    def test_settings_required_fields(self):
        """Test that settings have required fields."""
        from src.models.core import AppSettings

        settings = AppSettings()

        # These should exist and have reasonable defaults
        assert hasattr(settings, "chunk_size")
        assert hasattr(settings, "chunk_overlap")
        assert isinstance(settings.chunk_size, int)
        assert isinstance(settings.chunk_overlap, int)

        # Test additional key settings that should exist
        assert hasattr(settings, "dense_embedding_dimension")
        assert hasattr(settings, "dense_embedding_model")
        assert hasattr(settings, "retrieval_top_k")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
