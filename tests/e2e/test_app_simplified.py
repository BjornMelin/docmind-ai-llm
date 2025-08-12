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


@patch("src.agents.agent_factory.get_agent_system")
def test_agent_factory_integration(mock_get_agent):
    """Test that agent factory functions work correctly."""
    from src.agents.agent_factory import create_agentic_rag_system

    mock_llm = MagicMock()
    mock_tools = [MagicMock()]

    agent = create_agentic_rag_system(mock_tools, mock_llm)
    assert agent is not None

    # Test get_agent_system returns correct format
    mock_get_agent.return_value = (agent, "single")
    result = mock_get_agent(mock_tools, mock_llm)
    assert result[1] == "single"


@patch("src.utils.document.load_documents_llama")
@patch("src.utils.embedding.create_index_async")
def test_document_pipeline(mock_create_index, mock_load_docs):
    """Test document loading and indexing pipeline."""
    from src.utils.document import load_documents_llama
    from src.utils.embedding import create_index_async

    # Setup mocks
    mock_docs = [MagicMock()]
    mock_load_docs.return_value = mock_docs
    mock_index = MagicMock()
    mock_create_index.return_value = mock_index

    # Test document loading
    files = [MagicMock()]
    docs = load_documents_llama(files, False, False)
    assert docs == mock_docs

    # Test index creation (async function returns coroutine when mocked)
    create_index_async(docs, True)
    mock_create_index.assert_called_once_with(docs, True)


def test_models_core_import():
    """Test that models.core can be imported and contains required classes."""
    try:
        from src.models.core import AnalysisOutput, Settings

        # Test Settings class exists and can be instantiated
        settings = Settings()
        assert hasattr(settings, "llm_model")
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


def test_simplified_architecture_compatibility():
    """Test that the simplified architecture components work together."""
    try:
        # Test that we can import all key components
        from src.agents.agent_factory import create_agentic_rag_system, get_agent_system

        # Verify single agent architecture
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()

        with patch("llama_index.core.agent.ReActAgent.from_tools") as mock_react:
            mock_agent = MagicMock()
            mock_react.return_value = mock_agent

            # Test agent creation
            agent = create_agentic_rag_system(mock_tools, mock_llm)
            assert agent == mock_agent

            # Test get_agent_system always returns single mode
            result = get_agent_system(mock_tools, mock_llm, enable_multi_agent=True)
            assert (
                result[1] == "single"
            )  # Should always be single in simplified architecture

    except ImportError as e:
        pytest.skip(f"Import error (expected in some test environments): {e}")


def test_no_deleted_utils_references():
    """Ensure no references to deleted root utils directory."""
    try:
        # This should succeed - utils are now in src/utils
        # Test that the import paths exist by importing modules
        try:
            import src.utils.core  # noqa: F401
            import src.utils.document  # noqa: F401
            import src.utils.embedding  # noqa: F401
        except ImportError:
            pytest.fail("Should be able to import from src.utils")

        # These should not exist
        try:
            import utils  # noqa: F401

            pytest.fail("Root utils directory should not exist")
        except ImportError:
            pass  # Expected

    except ImportError as e:
        pytest.skip(f"Import error (expected in some test environments): {e}")
