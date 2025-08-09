"""Tests for Streamlit application functionality.

This module tests the main Streamlit application interface including hardware
detection, model selection, document upload/analysis, chat functionality, and
session persistence following 2025 best practices.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app_test():
    """Create an AppTest instance for testing the main application.

    Returns:
        AppTest: Streamlit app test instance.
    """
    return AppTest.from_file("app.py")


@patch("app.detect_hardware", return_value=("GPU detected", 16))
def test_app_hardware_detection(mock_detect, app_test):
    """Test hardware detection display in the application.

    Args:
        mock_detect: Mock hardware detection function.
        app_test: Streamlit app test fixture.
    """
    app = app_test.run()
    assert "Detected Hardware: GPU detected, VRAM: 16GB" in app.info[0].value


@patch("app.Ollama")
def test_app_model_selection(mock_ollama, app_test):
    """Test model selection functionality.

    Args:
        mock_ollama: Mock Ollama client.
        app_test: Streamlit app test fixture.
    """
    mock_ollama.return_value.invoke.return_value = "Test"
    app = app_test.run()
    app.selectbox[0].select("ollama")  # Backend
    app.run()
    assert not app.exception


def test_app_upload_and_analyze(app_test, tmp_path):
    """Test document upload and analysis workflow.

    Args:
        app_test: Streamlit app test fixture.
        tmp_path: Temporary directory for test files.
    """
    app = app_test.run()
    # Simulate upload
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF dummy")
    app.file_uploader[0].upload({"test.pdf": pdf.read_bytes()})
    app.button[2].click()  # Extract and Analyze
    app.run()
    assert "Analysis Results" in app.header


def test_app_chat_functionality(app_test, tmp_path):
    """Test chat functionality with document context.

    Args:
        app_test: Streamlit app test fixture.
        tmp_path: Temporary directory for test files.
    """
    app = app_test.run()
    # Setup vectorstore mock
    with patch("app.create_vectorstore") as mock_vs:
        mock_vs.return_value = MagicMock()
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF dummy")
        app.file_uploader[0].upload({"test.pdf": pdf.read_bytes()})
        app.button[2].click()  # Analyze
        app.text_input[1].input("Question?")  # Chat input
        app.button[3].click()  # Send
        app.run()
        assert "Assistant:" in app.markdown[-1].value


def test_app_session_persistence(app_test):
    """Test session save and load functionality.

    Args:
        app_test: Streamlit app test fixture.
    """
    app = app_test.run()
    app.button[0].click()  # Save
    app.button[1].click()  # Load
    app.run()
    assert "Session loaded" in app.success or "No saved session" in app.error
