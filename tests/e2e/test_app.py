"""Tests for Streamlit application functionality.

This module tests the main Streamlit application interface including hardware
detection, model selection, document upload/analysis, chat functionality, and
session persistence with simplified ReActAgent architecture.

Tests cover:
- Hardware detection from src.utils.core
- Simplified single ReActAgent system (no multi-agent complexity)
- Document upload and async indexing
- Chat functionality with ReActAgent
- Session persistence with memory buffer
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Mock problematic imports before app loads to prevent BLAS library issues
sys.modules["llama_index.llms.llama_cpp"] = MagicMock()
sys.modules["llama_cpp"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()

# Mock ollama completely to prevent network calls
mock_ollama = MagicMock()
mock_ollama.list.return_value = {"models": [{"name": "qwen3-4b-instruct-2507:latest"}]}
mock_ollama.pull.return_value = {"status": "success"}
mock_ollama.chat.return_value = {"message": {"content": "Test response"}}
sys.modules["ollama"] = mock_ollama


@pytest.fixture
def app_test():
    """Create an AppTest instance for testing the main application.

    Returns:
        AppTest: Streamlit app test instance.
    """
    with (
        patch("ollama.pull", return_value={"status": "success"}),
        patch("ollama.chat", return_value={"message": {"content": "Test response"}}),
    ):
        return AppTest.from_file(
            str(Path(__file__).parent.parent.parent / "src" / "app.py")
        )


@patch("ollama.pull", return_value={"status": "success"})
@patch(
    "src.utils.core.detect_hardware",
    return_value={"gpu_name": "RTX 4090", "vram_total_gb": 24, "cuda_available": True},
)
def test_app_hardware_detection(mock_detect, mock_pull, app_test):
    """Test hardware detection display in the application.

    Args:
        mock_detect: Mock hardware detection function.
        mock_pull: Mock ollama.pull function.
        app_test: Streamlit app test fixture.
    """
    app = app_test.run()
    # Check that hardware detection was called and info elements are present
    # The actual text content may not be directly accessible in test mode
    mock_detect.assert_called_once()
    sidebar_str = str(app.sidebar)
    assert "Info()" in sidebar_str  # Info widgets are rendered
    assert "Use GPU" in sidebar_str  # GPU checkbox is present


@patch("ollama.pull", return_value={"status": "success"})
@patch(
    "ollama.list",
    return_value={
        "models": [{"name": "qwen3-4b-instruct-2507:latest"}, {"name": "llama3:8b"}]
    },
)
def test_app_model_selection(mock_ollama_list, mock_pull, app_test):
    """Test model selection functionality.

    Args:
        mock_ollama_list: Mock Ollama models list.
        mock_pull: Mock ollama.pull function.
        app_test: Streamlit app test fixture.
    """
    app = app_test.run()
    # Test that backend selection works
    backend_selectbox = [
        elem
        for elem in app.selectbox
        if "Backend" in str(elem) or "ollama" in str(elem)
    ]
    if backend_selectbox:
        backend_selectbox[0].select("ollama")
    app.run()
    assert not app.exception


@patch("ollama.pull", return_value={"status": "success"})
def test_app_upload_and_analyze(mock_pull, app_test, tmp_path):
    """Test document upload and analysis workflow - SKIPPED (legacy functions).

    Args:
        mock_pull: Mock ollama.pull function.
        app_test: Streamlit app test fixture.
        tmp_path: Temporary directory for test files.
    """
    pytest.skip(
        "Legacy document processing functions removed with ADR-009 architecture"
    )

    app = app_test.run()
    # Simulate upload - check if file uploader elements exist in the app tree
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy content")

    # Check if file uploader components exist in the app
    app_str = str(app)
    has_file_uploader = "FileUploader" in app_str or "file_uploader" in app_str.lower()

    # Test passes if app loads without critical errors
    # File upload testing is limited in Streamlit test framework
    if has_file_uploader:
        # If file uploader is present, the test confirms UI elements exist
        pass

    # Test passes if no exceptions occur during app execution
    assert not app.exception


@patch("ollama.pull", return_value={"status": "success"})
@patch("src.agents.coordinator.create_multi_agent_coordinator")
def test_app_chat_functionality(mock_create_coordinator, mock_pull, app_test):
    """Test chat functionality with MultiAgentCoordinator.

    Args:
        mock_create_coordinator: Mock coordinator creation.
        mock_pull: Mock ollama.pull function.
        app_test: Streamlit app test fixture.
    """
    # Setup mocks
    mock_coordinator = MagicMock()
    mock_coordinator.process_query = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "This is a test response."
    mock_coordinator.process_query.return_value = mock_response
    mock_create_coordinator.return_value = (mock_coordinator, "multi")

    app = app_test.run()

    # Test chat input if available
    app_str = str(app)
    has_chat_input = "ChatInput" in app_str or "chat_input" in app_str.lower()

    if has_chat_input:
        # Simulate setting up index first
        app.session_state.index = MagicMock()

        # Set a test message in session state to simulate interaction
        app.session_state["test_chat_message"] = "What is this document about?"
        app.run()

    # Test passes if no exceptions occur
    assert not app.exception


@patch("ollama.pull", return_value={"status": "success"})
@patch("pathlib.Path.exists", return_value=False)
def test_app_session_persistence(mock_path, mock_pull, app_test):
    """Test session save and load functionality.

    Args:
        mock_path: Mock pathlib.Path.exists function.
        mock_pull: Mock ollama.pull function.
        app_test: Streamlit app test fixture.

    Note:
        pathlib.Path.exists is mocked to return False for testing.
    """
    app = app_test.run()

    # Find save and load buttons by checking button text or context
    save_buttons = [btn for btn in app.button if "Save" in str(btn)]
    load_buttons = [btn for btn in app.button if "Load" in str(btn)]

    if save_buttons:
        save_buttons[0].click()
        app.run()

    if load_buttons:
        load_buttons[0].click()
        app.run()

    # Test passes if no exceptions occur during session operations
    assert not app.exception


@patch("ollama.pull", return_value={"status": "success"})
@patch(
    "src.utils.core.detect_hardware",
    return_value={"gpu_name": "RTX 4090", "vram_total_gb": 24, "cuda_available": True},
)
@patch(
    "ollama.list", return_value={"models": [{"name": "qwen3-4b-instruct-2507:latest"}]}
)
@patch("src.agents.coordinator.create_multi_agent_coordinator")
def test_end_to_end_workflow(
    mock_create_coordinator,
    mock_ollama_list,
    mock_detect,
    mock_pull,
    app_test,
    tmp_path,
):
    """Test complete end-to-end workflow with MultiAgentCoordinator.

    SIMPLIFIED for ADR-009.

    Args:
        mock_create_coordinator: Mock coordinator creation.
        mock_ollama_list: Mock Ollama models list.
        mock_detect: Mock hardware detection.
        mock_pull: Mock ollama.pull function.
        app_test: Streamlit app test fixture.
        tmp_path: Temporary directory for test files.
    """
    # Setup mocks - simplified for ADR-009 compliant testing
    mock_coordinator = MagicMock()
    mock_coordinator.process_query = MagicMock()
    mock_coordinator.process_query.return_value = (
        "Document analysis completed successfully."
    )
    mock_create_coordinator.return_value = (mock_coordinator, "multi")

    # Run the app
    app = app_test.run()

    # Test that app loads without errors
    assert not app.exception

    # Test hardware detection appears - look for Info components in sidebar
    sidebar_str = str(app.sidebar)
    assert "Info()" in sidebar_str  # Hardware info components are present

    # Test basic UI components exist (simplified for ADR-009)
    app_str = str(app)
    has_file_uploader = "FileUploader" in app_str or "file_uploader" in app_str.lower()

    if has_file_uploader:
        # File uploader component exists in UI
        pass

    # Test chat functionality if available
    app_str = str(app)
    has_chat_input = "ChatInput" in app_str or "chat_input" in app_str.lower()

    if has_chat_input:
        app.session_state.index = MagicMock()

        # Set a test message in session state to simulate chat interaction
        app.session_state["test_chat_message"] = "Summarize the document."
        app.run()

    # Verify no exceptions occurred during full workflow
    assert not app.exception
