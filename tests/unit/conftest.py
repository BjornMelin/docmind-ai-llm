"""Shared fixtures for document loader tests.

This module provides common fixtures for all document loader test modules
following 2025 best practices with proper typing and pytest-mock usage.
"""

import base64
import io
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from PIL import Image

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.core import AppSettings


@pytest.fixture
def mock_settings() -> AppSettings:
    """Create mock settings for testing."""
    return AppSettings(
        parse_strategy="hi_res",
        chunk_size=1024,
        chunk_overlap=200,
        dense_embedding_model="BAAI/bge-large-en-v1.5",
    )


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> str:
    """Create a temporary PDF file for testing."""
    pdf_path = tmp_path / "test.pdf"
    # Create minimal PDF content
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
180
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return str(pdf_path)


@pytest.fixture
def sample_image_base64() -> str:
    """Create a sample base64-encoded image for testing."""
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
    """Create a mock fitz.Document with proper spec."""
    try:
        import fitz

        return mocker.Mock(spec=fitz.Document)
    except ImportError:
        # Fallback for when PyMuPDF is not available
        return mocker.Mock()


@pytest.fixture
def mock_fitz_page(mocker) -> Mock:
    """Create a mock fitz.Page with proper spec."""
    try:
        import fitz

        return mocker.Mock(spec=fitz.Page)
    except ImportError:
        # Fallback for when PyMuPDF is not available
        return mocker.Mock()


@pytest.fixture
def mock_fitz_pixmap(mocker) -> Mock:
    """Create a mock fitz.Pixmap with proper spec."""
    try:
        import fitz

        return mocker.Mock(spec=fitz.Pixmap)
    except ImportError:
        # Fallback for when PyMuPDF is not available
        return mocker.Mock()


@pytest.fixture
def mock_unstructured_partition(mocker) -> Mock:
    """Create a mock for unstructured partition function."""
    return mocker.Mock()


@pytest.fixture
def mock_model_manager(mocker) -> Mock:
    """Create a mock ModelManager instance."""
    return mocker.Mock()


@pytest.fixture
def sample_text_file(tmp_path: Path) -> str:
    """Create a sample text file for testing."""
    text_path = tmp_path / "test.txt"
    text_path.write_text("This is a test document for loading.")
    return str(text_path)


@pytest.fixture
def sample_docx_file(tmp_path: Path) -> str:
    """Create a sample DOCX file for testing."""
    docx_path = tmp_path / "test.docx"
    # Create minimal DOCX structure (this is a simplified version)
    docx_path.write_bytes(b"PK\x03\x04" + b"\x00" * 26 + b"sample docx content")
    return str(docx_path)


@pytest.fixture
def mock_uploaded_file() -> Mock:
    """Create a mock uploaded file for testing."""
    mock_file = Mock()
    mock_file.name = "test_document.pdf"
    mock_file.type = "application/pdf"
    mock_file.getvalue.return_value = b"mock pdf content"
    return mock_file
