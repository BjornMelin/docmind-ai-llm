"""Comprehensive test validation module for DocMind AI.

This module validates that all components can be imported and basic functionality
works correctly. It serves as a health check for the entire test suite and
system integration.
"""

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import settings


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
                    assert hasattr(module, "UnstructuredChunker")
                elif module_name == "src.cache.simple_cache":
                    assert hasattr(module, "SimpleCache")
                elif module_name == "src.processing.embeddings.bgem3_embedder":
                    assert hasattr(module, "BGEM3Embedder")

            except ImportError as e:
                pytest.fail(f"Failed to import ADR-009 module {module_name}: {e}")

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
        from src.utils.core import validate_startup_configuration

        # validate_startup_configuration now requires a settings parameter
        try:
            result = validate_startup_configuration(settings)
            # Result should be a dict with validation results
            assert isinstance(result, dict)
            assert "valid" in result
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            pytest.fail(f"validate_startup_configuration failed: {e}")

    def test_key_file_structure(self):
        """Test that essential files exist in correct locations."""
        base_path = Path(__file__).parent.parent.parent

        required_files = [
            "src/models/core.py",
            "src/agents/tool_factory.py",
            "src/utils/core.py",
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
            from src.utils.core import detect_hardware

            # Create basic settings instance
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
        assert settings is not None
        # Check for actual attributes that exist in the simplified Settings class
        assert hasattr(settings, "qdrant_url")
        assert hasattr(settings, "enable_gpu_acceleration")
        assert hasattr(settings, "chunk_size")

    def test_settings_required_fields(self):
        """Test that settings have required fields."""
        # These should exist and have reasonable defaults
        assert hasattr(settings, "chunk_size")
        assert hasattr(settings, "chunk_overlap")
        assert isinstance(settings.chunk_size, int)
        assert isinstance(settings.chunk_overlap, int)

        # Test additional key settings that should exist
        assert hasattr(settings, "embedding_dimension")
        assert hasattr(settings, "embedding_model")
        assert hasattr(settings, "top_k")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
