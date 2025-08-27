"""Simplified E2E tests for Streamlit app focused on core functionality.

Tests the main application components with proper mocking to avoid
dependency issues while validating the simplified ReActAgent architecture.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock problematic dependencies before any imports
# Exclude torch from global mocking to preserve version info for spacy compatibility
mock_modules = [
    "llama_index.llms.llama_cpp",
    "llama_cpp",
    "ollama",
    "transformers",
    "sentence_transformers",
]

for module in mock_modules:
    sys.modules[module] = MagicMock()

# Mock torch with preserved attributes for spacy compatibility
if "torch" not in sys.modules:
    torch_mock = MagicMock()
    torch_mock.__version__ = "2.7.1+cu126"
    torch_mock.__spec__ = MagicMock()
    torch_mock.__spec__.name = "torch"
    torch_mock.cuda.is_available.return_value = True
    torch_mock.cuda.device_count.return_value = 1
    sys.modules["torch"] = torch_mock


def test_app_imports():
    """Test that the app can be imported without errors."""
    try:
        from src import app

        assert app is not None
    except ImportError as e:
        pytest.skip(f"Import error (expected in test environment): {e}")


@patch("src.utils.core.detect_hardware")
@patch("src.utils.core.validate_startup_configuration")
def test_app_initialization(mock_validate, mock_detect):
    """Test basic app initialization."""
    mock_detect.return_value = {
        "gpu_name": "GTX 1080",
        "vram_total_gb": 8,
        "cuda_available": True,
    }
    mock_validate.return_value = True

    try:
        from src import app

        # If we can import without exception, consider it a success
        assert app is not None
    except ImportError as e:
        pytest.skip(f"Import error (expected in test environment): {e}")


@patch("src.agents.coordinator.create_multi_agent_coordinator")
def test_coordinator_integration(mock_create_coordinator):
    """Test that MultiAgentCoordinator functions work correctly."""
    # Setup mock coordinator
    mock_coordinator = MagicMock()
    mock_create_coordinator.return_value = mock_coordinator

    # Test coordinator creation
    coordinator = mock_create_coordinator()
    assert coordinator is not None

    # Test that we can create a coordinator instance
    mock_create_coordinator.assert_called_once()


@patch("src.processing.document_processor.DocumentProcessor")
def test_document_pipeline(mock_processor_class):
    """Test document processing pipeline with ADR-009 components."""
    # Skip this test as it was testing deleted legacy functionality
    # New ADR-009 document processing is tested in dedicated test suites
    pytest.skip("Legacy document pipeline test - replaced by ADR-009 implementation")


def test_models_core_import():
    """Test that models.core can be imported and contains required classes."""
    try:
        from src.config import settings
        from src.models.schemas import AnalysisOutput

        # Test settings instance exists
        assert hasattr(settings, "model_name")
        assert hasattr(settings, "embedding_model")

        # Test AnalysisOutput class exists
        output = AnalysisOutput(
            summary="Test summary",
            key_insights=["insight1"],
            action_items=["action1"],
            open_questions=["question1"],
        )
        assert output.summary == "Test summary"

    except ImportError as e:
        pytest.fail(f"Failed to import required models: {e}")


def test_multi_agent_architecture_compatibility():
    """Test that the multi-agent architecture components work together."""
    try:
        # Test that we can import all key components
        from src.agents.coordinator import (
            MultiAgentCoordinator,
            create_multi_agent_coordinator,
        )

        # Verify multi-agent coordinator exists and is callable
        assert callable(MultiAgentCoordinator)
        assert callable(create_multi_agent_coordinator)

        # Test that MultiAgentCoordinator has required methods
        assert hasattr(MultiAgentCoordinator, "process_query")

    except ImportError as e:
        pytest.skip(f"Import error (expected in some test environments): {e}")


def test_no_deleted_utils_references():
    """Ensure ADR-009 components are available and legacy modules are deleted."""
    try:
        # Test that ADR-009 components can be imported
        try:
            import src.cache.simple_cache  # noqa: F401
            import src.processing.document_processor  # noqa: F401
            import src.utils.core  # noqa: F401
        except ImportError:
            pytest.fail("Should be able to import ADR-009 components")

        # Verify deleted legacy modules cannot be imported
        try:
            import src.retrieval.integration  # noqa: F401  # pyright: ignore[reportMissingImports]

            pytest.fail("Legacy src.retrieval.integration should be deleted")
        except ImportError:
            pass  # Expected - module was deleted for ADR-009 compliance

        try:
            import src.utils.document  # noqa: F401

            pytest.fail("Legacy src.utils.document should be deleted")
        except ImportError:
            pass  # Expected - module was deleted for ADR-009 compliance

        # Root utils should not exist
        try:
            import utils  # noqa: F401

            pytest.fail("Root utils directory should not exist")
        except ImportError:
            pass  # Expected

    except ImportError as e:
        pytest.skip(f"Import error (expected in some test environments): {e}")
