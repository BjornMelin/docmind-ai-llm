"""Unit test fixtures for document processing and utilities.

Provides specialized fixtures for unit-level testing with focus on:
- Document processing mock factories
- Image processing utilities
- File format testing helpers
- Lightweight mock objects for fast unit tests

Follows 2025 pytest patterns with proper typing and async support.
"""

import base64
import io
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from PIL import Image

# Import path handling moved to pytest.ini pythonpath configuration

# Note: mock_settings removed - use fixtures from main conftest.py
# Note: PDF creation moved to use temp_pdf_file from main conftest.py


@pytest.fixture
def sample_image_base64() -> str:
    """Create a sample base64-encoded image for testing.

    Returns:
        str: Base64 encoded PNG image for testing image processing.
    """
    # Create a small test image
    img = Image.new("RGB", (10, 10), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def sample_images_data(sample_image_base64: str) -> list[dict[str, Any]]:
    """Create sample image data for testing."""
    return [
        {
            "image_base64": sample_image_base64,
            "image_mimetype": "image/png",
            "image_path": "test_image.png",
        }
    ]


@pytest.fixture
def mock_fitz_document(mocker) -> Mock:
    """Create a mock fitz.Document with proper spec.

    Returns:
        Mock: Mock PyMuPDF Document with proper interface.
    """
    try:
        import fitz

        return mocker.Mock(spec=fitz.Document)
    except ImportError:
        # Fallback for when PyMuPDF is not available
        return mocker.Mock()


@pytest.fixture
def mock_fitz_page(mocker) -> Mock:
    """Create a mock fitz.Page with proper spec.

    Returns:
        Mock: Mock PyMuPDF Page with proper interface.
    """
    try:
        import fitz

        return mocker.Mock(spec=fitz.Page)
    except ImportError:
        # Fallback for when PyMuPDF is not available
        return mocker.Mock()


@pytest.fixture
def mock_fitz_pixmap(mocker) -> Mock:
    """Create a mock fitz.Pixmap with proper spec.

    Returns:
        Mock: Mock PyMuPDF Pixmap with proper interface.
    """
    try:
        import fitz

        return mocker.Mock(spec=fitz.Pixmap)
    except ImportError:
        # Fallback for when PyMuPDF is not available
        return mocker.Mock()


@pytest.fixture
def mock_unstructured_partition(mocker) -> Mock:
    """Create a mock for unstructured partition function.

    Returns:
        Mock: Mock unstructured partition function for document processing.
    """
    return mocker.Mock()


@pytest.fixture
def mock_model_manager(mocker) -> Mock:
    """Create a mock ModelManager instance.

    Returns:
        Mock: Mock ModelManager for testing model lifecycle.
    """
    return mocker.Mock()


@pytest.fixture
def sample_text_file(tmp_path: Path) -> str:
    """Create a sample text file for testing.

    Returns:
        str: Path to temporary text file with test content.
    """
    text_path = tmp_path / "test.txt"
    text_path.write_text("This is a test document for loading.")
    return str(text_path)


@pytest.fixture
def sample_docx_file(tmp_path: Path) -> str:
    """Create a sample DOCX file for testing.

    Returns:
        str: Path to minimal DOCX file for format testing.
    """
    docx_path = tmp_path / "test.docx"
    # Create minimal DOCX structure (simplified for testing)
    docx_path.write_bytes(b"PK\x03\x04" + b"\x00" * 26 + b"sample docx content")
    return str(docx_path)


@pytest.fixture
def mock_uploaded_file() -> Mock:
    """Create a mock uploaded file for testing.

    Returns:
        Mock: Mock uploaded file with standard interface for Streamlit file uploads.
    """
    mock_file = Mock()
    mock_file.name = "test_document.pdf"
    mock_file.type = "application/pdf"
    mock_file.getvalue.return_value = b"mock pdf content"
    return mock_file
