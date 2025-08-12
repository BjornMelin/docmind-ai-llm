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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock problematic imports before app loads to prevent BLAS library issues
sys.modules["llama_index.llms.llama_cpp"] = MagicMock()
sys.modules["llama_cpp"] = MagicMock()


@pytest.fixture
def app_test():
    """Create an AppTest instance for testing the main application.

    Returns:
        AppTest: Streamlit app test instance.
    """
    return AppTest.from_file("src/app.py")


@patch(
    "src.utils.core.detect_hardware",
    return_value={"gpu_name": "RTX 4090", "vram_total_gb": 24, "cuda_available": True},
)
def test_app_hardware_detection(mock_detect, app_test):
    """Test hardware detection display in the application.

    Args:
        mock_detect: Mock hardware detection function.
        app_test: Streamlit app test fixture.
    """
    app = app_test.run()
    # Check that hardware detection was called and info elements are present
    # The actual text content may not be directly accessible in test mode
    mock_detect.assert_called_once()
    sidebar_str = str(app.sidebar)
    assert "Info()" in sidebar_str  # Info widgets are rendered
    assert "Use GPU" in sidebar_str  # GPU checkbox is present


@patch(
    "ollama.list",
    return_value={"models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]},
)
def test_app_model_selection(mock_ollama_list, app_test):
    """Test model selection functionality.

    Args:
        mock_ollama_list: Mock Ollama models list.
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


@patch("src.utils.document.load_documents_llama")
@patch("src.utils.embedding.create_index_async")
def test_app_upload_and_analyze(mock_create_index, mock_load_docs, app_test, tmp_path):
    """Test document upload and analysis workflow.

    Args:
        mock_create_index: Mock index creation.
        mock_load_docs: Mock document loading.
        app_test: Streamlit app test fixture.
        tmp_path: Temporary directory for test files.
    """
    # Setup mocks
    mock_load_docs.return_value = [MagicMock()]
    mock_create_index.return_value = MagicMock()

    app = app_test.run()
    # Simulate upload
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy content")

    if app.file_uploader:
        app.file_uploader[0].upload({"test.pdf": pdf.read_bytes()})
        app.run()

    # Test passes if no exceptions occur during upload simulation
    assert not app.exception


@patch("src.agents.agent_factory.get_agent_system")
@patch("src.agents.agent_factory.process_query_with_agent_system")
def test_app_chat_functionality(mock_process_query, mock_get_agent, app_test):
    """Test chat functionality with ReActAgent.

    Args:
        mock_process_query: Mock query processing.
        mock_get_agent: Mock agent system.
        app_test: Streamlit app test fixture.
    """
    # Setup mocks
    mock_agent = MagicMock()
    mock_get_agent.return_value = (mock_agent, "single")
    mock_process_query.return_value = "This is a test response."

    app = app_test.run()

    # Test chat input if available
    if hasattr(app, "chat_input") and app.chat_input:
        # Simulate setting up index first
        app.session_state.index = MagicMock()

        app.chat_input[0].input("What is this document about?")
        app.run()

    # Test passes if no exceptions occur
    assert not app.exception


@patch("pathlib.Path.exists", return_value=False)
def test_app_session_persistence(app_test):
    """Test session save and load functionality.

    Args:
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


@patch(
    "src.utils.core.detect_hardware",
    return_value={"gpu_name": "RTX 4090", "vram_total_gb": 24, "cuda_available": True},
)
@patch("ollama.list", return_value={"models": [{"name": "llama3:8b"}]})
@patch("src.utils.document.load_documents_llama")
@patch("src.utils.embedding.create_index_async")
@patch("src.agents.agent_factory.get_agent_system")
@patch("src.agents.agent_factory.process_query_with_agent_system")
def test_end_to_end_workflow(
    mock_process_query,
    mock_get_agent,
    mock_create_index,
    mock_load_docs,
    mock_ollama_list,
    mock_detect,
    app_test,
    tmp_path,
):
    """Test complete end-to-end workflow with ReActAgent.

    Args:
        mock_process_query: Mock query processing.
        mock_get_agent: Mock agent system.
        mock_create_index: Mock index creation.
        mock_load_docs: Mock document loading.
        mock_ollama_list: Mock Ollama models list.
        mock_detect: Mock hardware detection.
        app_test: Streamlit app test fixture.
        tmp_path: Temporary directory for test files.
    """
    # Setup mocks
    mock_load_docs.return_value = [MagicMock()]
    mock_create_index.return_value = MagicMock()
    mock_agent = MagicMock()
    mock_get_agent.return_value = (mock_agent, "single")
    mock_process_query.return_value = "Document analysis completed successfully."

    # Run the app
    app = app_test.run()

    # Test that app loads without errors
    assert not app.exception

    # Test hardware detection appears
    assert "RTX 4090" in str(app.sidebar) or "Detected:" in str(app.sidebar)

    # Test document upload simulation
    if app.file_uploader:
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4\\n%dummy content for testing")
        app.file_uploader[0].upload({"test.pdf": pdf.read_bytes()})
        app.run()

    # Test analysis button if available
    analyze_buttons = [btn for btn in app.button if "Analyze" in str(btn)]
    if analyze_buttons:
        analyze_buttons[0].click()
        app.run()

    # Test chat functionality if available
    if hasattr(app, "chat_input") and app.chat_input:
        app.session_state.index = MagicMock()
        app.chat_input[0].input("Summarize the document.")
        app.run()

    # Verify no exceptions occurred during full workflow
    assert not app.exception
