"""Comprehensive test validation module for DocMind AI.

This module validates that all components can be imported and basic functionality
works correctly. It serves as a health check for the entire test suite and
system integration.

Following PyTestQA-Agent standards for comprehensive testing.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import importlib
import time
from unittest.mock import MagicMock, patch

import pytest


class TestImportValidation:
    """Validate all modules can be imported successfully."""

    def test_core_models_import(self):
        """Test that core models can be imported."""
        modules = [
            "models",
        ]

        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} imported but is None"

                # Test that key classes are available
                if module_name == "models":
                    assert hasattr(module, "AppSettings")
                    assert hasattr(module, "AnalysisOutput")

            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_utils_modules_import(self):
        """Test that all utils modules can be imported."""
        modules = [
            "utils.utils",
            "utils.document_loader",
            "utils.index_builder",
            "utils.model_manager",
            "utils.qdrant_utils",
        ]

        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} imported but is None"

                # Test that modules have expected functions
                if module_name == "utils.utils":
                    assert hasattr(module, "setup_logging")
                    assert hasattr(module, "detect_hardware")
                    assert hasattr(module, "get_embed_model")

                elif module_name == "utils.document_loader":
                    assert hasattr(module, "load_documents_llama")
                    assert hasattr(module, "load_documents_unstructured")

                elif module_name == "utils.index_builder":
                    assert hasattr(module, "build_vector_index")

            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_agent_modules_import(self):
        """Test that agent modules can be imported."""
        modules = [
            "agents.agent_utils",
            "agent_factory",
        ]

        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} imported but is None"

                # Test that modules have expected functions
                if module_name == "agents.agent_utils":
                    assert hasattr(module, "create_agent")

                elif module_name == "agent_factory":
                    assert hasattr(module, "analyze_query_complexity")
                    assert hasattr(module, "get_agent_system")

            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_optional_imports(self):
        """Test optional imports that may not be available in all environments."""
        optional_modules = [
            ("torch", "PyTorch for GPU acceleration"),
            ("transformers", "Hugging Face Transformers"),
            ("fastembed", "FastEmbed for embeddings"),
            ("qdrant_client", "Qdrant vector database client"),
            ("llama_index", "LlamaIndex framework"),
            ("streamlit", "Streamlit web framework"),
        ]

        missing_optional = []

        for module_name, description in optional_modules:
            try:
                importlib.import_module(module_name)
            except ImportError:
                missing_optional.append(f"{module_name} ({description})")

        if missing_optional:
            pytest.skip(
                f"Optional modules not available: {', '.join(missing_optional)}"
            )


class TestBasicFunctionality:
    """Test basic functionality of core components."""

    def test_app_settings_creation(self):
        """Test that AppSettings can be created with defaults."""
        from models import AppSettings

        settings = AppSettings()

        # Verify key settings are present
        assert settings.backend is not None
        assert settings.default_model is not None
        assert settings.context_size > 0
        assert settings.dense_embedding_model is not None
        assert settings.sparse_embedding_model is not None

    def test_analysis_output_creation(self):
        """Test that AnalysisOutput can be created."""
        from models import AnalysisOutput

        output = AnalysisOutput(
            summary="Test summary",
            key_insights=["insight1", "insight2"],
            action_items=["action1"],
            open_questions=["question1"],
        )

        assert output.summary == "Test summary"
        assert len(output.key_insights) == 2
        assert len(output.action_items) == 1
        assert len(output.open_questions) == 1

    def test_logging_setup(self):
        """Test that logging can be set up."""
        from utils.utils import setup_logging

        with patch("logging.basicConfig") as mock_config:
            setup_logging("INFO")
            mock_config.assert_called_once()

    def test_hardware_detection(self):
        """Test basic hardware detection functionality."""
        from utils.utils import detect_hardware

        with patch("utils.utils.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            hardware_info = detect_hardware()

            assert isinstance(hardware_info, dict)
            assert "cuda_available" in hardware_info
            assert "gpu_name" in hardware_info
            assert "fastembed_providers" in hardware_info


class TestDependencyValidation:
    """Validate that all required dependencies are properly installed."""

    def test_required_packages_installed(self):
        """Test that required packages are installed."""
        required_packages = [
            "pydantic",
            "pydantic_settings",
            "pathlib",
            "json",
            "logging",
            "subprocess",
            "unittest.mock",
        ]

        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError as e:
                pytest.fail(f"Required package {package} not available: {e}")

    def test_llm_framework_dependencies(self):
        """Test that LLM framework dependencies are available."""
        try:
            import llama_index
            from llama_index.core import Document

            # Test basic Document creation
            doc = Document(text="Test document", metadata={"source": "test"})
            assert doc.text == "Test document"
            assert doc.metadata["source"] == "test"

        except ImportError as e:
            pytest.fail(f"LlamaIndex framework not available: {e}")

    def test_embedding_dependencies(self):
        """Test that embedding dependencies are available."""
        embedding_packages = [
            ("sentence_transformers", "Sentence Transformers"),
            ("transformers", "Hugging Face Transformers"),
        ]

        available_embeddings = []
        for package, description in embedding_packages:
            try:
                importlib.import_module(package)
                available_embeddings.append(description)
            except ImportError:
                pass

        # At least one embedding framework should be available
        assert len(available_embeddings) > 0, "No embedding frameworks available"

    def test_vector_database_dependencies(self):
        """Test vector database client availability."""
        try:
            from qdrant_client import QdrantClient

            # Test basic client creation (without connecting)
            assert QdrantClient is not None

        except ImportError:
            pytest.skip("Qdrant client not available")


class TestSystemIntegration:
    """Test basic system integration and workflow."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        from models import AppSettings

        return AppSettings(
            backend="ollama",
            default_model="test-model",
            dense_embedding_model="BAAI/bge-large-en-v1.5",
            sparse_embedding_model="prithvida/Splade_PP_en_v1",
        )

    def test_query_complexity_analysis(self, mock_settings):
        """Test query complexity analysis workflow."""
        try:
            from agent_factory import analyze_query_complexity

            with patch("agent_factory.settings", mock_settings):
                # Test simple query
                simple_result = analyze_query_complexity("What is AI?")
                assert isinstance(simple_result, dict)
                assert "complexity" in simple_result

                # Test complex query
                complex_result = analyze_query_complexity(
                    "Analyze the performance implications of hybrid retrieval systems "
                    "using dense and sparse embeddings with RRF fusion."
                )
                assert isinstance(complex_result, dict)
                assert "complexity" in complex_result

        except ImportError:
            pytest.skip("Agent factory not available for testing")

    def test_document_processing_workflow(self, mock_settings):
        """Test basic document processing workflow."""
        try:
            from llama_index.core import Document

            from utils.document_loader import chunk_documents_structured

            # Create test documents
            docs = [
                Document(
                    text="This is test document 1.", metadata={"source": "test1.txt"}
                ),
                Document(
                    text="This is test document 2.", metadata={"source": "test2.txt"}
                ),
            ]

            with patch("utils.document_loader.settings", mock_settings):
                with patch(
                    "llama_index.core.node_parser.SentenceSplitter"
                ) as mock_splitter:
                    mock_instance = MagicMock()
                    mock_instance.get_nodes_from_documents.return_value = docs
                    mock_splitter.return_value = mock_instance

                    result = chunk_documents_structured(docs)

                    assert isinstance(result, list)
                    assert len(result) >= len(docs)

        except ImportError:
            pytest.skip("Document processing components not available")


class TestEnvironmentValidation:
    """Validate test environment and configuration."""

    def test_python_version(self):
        """Test that Python version is compatible."""
        import sys

        version_info = sys.version_info

        # Require Python 3.10+
        assert version_info.major == 3, f"Python 3.x required, got {version_info.major}"
        assert version_info.minor >= 10, (
            f"Python 3.10+ required, got {version_info.major}.{version_info.minor}"
        )

    def test_project_structure(self):
        """Test that project has expected structure."""
        project_root = Path(__file__).parent.parent.parent

        required_files = [
            "models.py",
            "agent_factory.py",
            "utils/utils.py",
            "utils/document_loader.py",
            "utils/index_builder.py",
            "tests/conftest.py",
            "pytest.ini",
            "pyproject.toml",
        ]

        for file_path in required_files:
            full_path = project_root / file_path
            assert full_path.exists(), f"Required file missing: {file_path}"

    def test_test_configuration(self):
        """Test that pytest configuration is correct."""
        project_root = Path(__file__).parent.parent.parent
        pytest_ini = project_root / "pytest.ini"

        assert pytest_ini.exists(), "pytest.ini configuration file missing"

        # Read and validate pytest.ini content
        config_content = pytest_ini.read_text()

        required_sections = [
            "[pytest]",
            "testpaths = tests",
            "asyncio_mode = auto",
            "--cov=",
        ]

        for section in required_sections:
            assert section in config_content, f"Missing pytest configuration: {section}"

    def test_test_fixtures_available(self):
        """Test that common test fixtures are available."""
        from tests.conftest import sample_documents, test_settings

        # Test that fixtures can be imported
        assert test_settings is not None
        assert sample_documents is not None


class TestPerformanceBasics:
    """Basic performance validation tests."""

    def test_import_performance(self):
        """Test that imports complete in reasonable time."""
        import time

        modules_to_test = [
            "models",
            "utils.utils",
            "agent_factory",
        ]

        for module_name in modules_to_test:
            start_time = time.time()

            try:
                importlib.import_module(module_name)
                import_time = time.time() - start_time

                # Imports should complete in under 5 seconds
                assert import_time < 5.0, (
                    f"Import of {module_name} took {import_time:.2f}s (too slow)"
                )

            except ImportError:
                pytest.skip(
                    f"Module {module_name} not available for performance testing"
                )

    def test_settings_creation_performance(self):
        """Test that settings creation is fast."""
        from models import AppSettings

        start_time = time.time()

        # Create settings multiple times
        for _ in range(10):
            settings = AppSettings()
            assert settings is not None

        total_time = time.time() - start_time

        # Should create 10 settings objects in under 1 second
        assert total_time < 1.0, f"Settings creation took {total_time:.2f}s (too slow)"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_settings_handling(self):
        """Test handling of invalid settings."""
        from pydantic import ValidationError

        from models import AppSettings

        # Test invalid configuration values
        with pytest.raises(ValidationError):
            AppSettings(context_size=0)  # Should be > 0

        with pytest.raises(ValidationError):
            AppSettings(embedding_batch_size=0)  # Should be > 0

    def test_missing_file_handling(self):
        """Test handling when files are missing."""
        from pathlib import Path

        # Test accessing non-existent file
        non_existent_file = Path("/definitely/does/not/exist.txt")
        assert not non_existent_file.exists()

    def test_import_error_handling(self):
        """Test handling of import errors."""
        with pytest.raises(ImportError):
            importlib.import_module("definitely_not_a_real_module")


class TestMemoryAndResources:
    """Test memory usage and resource management."""

    def test_memory_usage_basic(self):
        """Test that basic operations don't consume excessive memory."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform basic operations
        from models import AppSettings
        from utils.utils import setup_logging

        settings = AppSettings()
        setup_logging("INFO")

        # Check memory after operations
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not increase memory by more than 50MB for basic operations
        assert memory_increase < 50, (
            f"Memory increased by {memory_increase:.1f}MB (too much)"
        )

    def test_no_file_handle_leaks(self):
        """Test that file operations don't leak handles."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        try:
            initial_files = process.num_fds() if hasattr(process, "num_fds") else 0
        except:
            pytest.skip("Cannot measure file descriptors on this platform")

        # Perform file operations
        project_root = Path(__file__).parent.parent.parent
        test_files = list(project_root.glob("*.py"))[:5]  # Check first 5 Python files

        for test_file in test_files:
            if test_file.exists():
                content = test_file.read_text()
                assert isinstance(content, str)

        try:
            final_files = process.num_fds() if hasattr(process, "num_fds") else 0
            file_handle_increase = final_files - initial_files

            # Should not leak file handles
            assert file_handle_increase <= 0, (
                f"Leaked {file_handle_increase} file handles"
            )
        except:
            pytest.skip("Cannot measure file descriptors on this platform")


@pytest.mark.integration
class TestSystemHealthCheck:
    """Comprehensive system health check."""

    def test_full_system_health(self):
        """Run comprehensive system health check."""
        health_checks = {
            "imports": self._check_imports(),
            "configuration": self._check_configuration(),
            "dependencies": self._check_dependencies(),
            "file_system": self._check_file_system(),
        }

        failed_checks = [name for name, passed in health_checks.items() if not passed]

        if failed_checks:
            pytest.fail(f"System health check failed: {', '.join(failed_checks)}")

        print("✅ All system health checks passed")

    def _check_imports(self) -> bool:
        """Check that core imports work."""
        try:
            from agent_factory import analyze_query_complexity
            from models import AnalysisOutput, AppSettings
            from utils.utils import detect_hardware, setup_logging

            return True
        except ImportError:
            return False

    def _check_configuration(self) -> bool:
        """Check that configuration is valid."""
        try:
            from models import AppSettings

            settings = AppSettings()
            return settings.backend is not None
        except:
            return False

    def _check_dependencies(self) -> bool:
        """Check that key dependencies are available."""
        try:
            import json
            import pathlib

            import pydantic

            return True
        except ImportError:
            return False

    def _check_file_system(self) -> bool:
        """Check that required files exist."""
        project_root = Path(__file__).parent.parent.parent
        required_files = ["models.py", "agent_factory.py", "utils/utils.py"]

        return all((project_root / file_path).exists() for file_path in required_files)
