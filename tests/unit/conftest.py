"""Unit test fixtures for document processing and utilities.

Provides specialized fixtures for unit-level testing with focus on:
- Document processing mock factories
- Image processing utilities
- File format testing helpers
- Lightweight mock objects for fast unit tests

Follows 2025 pytest patterns with proper typing and async support.
"""

import asyncio as _asyncio
import base64
import io
import time as _time
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from src.config.settings import DocMindSettings

# pylint: disable=redefined-outer-name
# Rationale: pytest fixture dependency injection intentionally uses argument names
# that shadow fixture factory functions defined at module scope. Renaming breaks
# fixture resolution; suppress at module level for tests.

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


# --- Global stabilization fixtures for unit tier ---


@pytest.fixture(autouse=True)
def no_sleep(request):
    """Avoid real delays: patch sleeps globally, but not for timeout tests.

    Some tests rely on real timeouts (e.g., thread-pool futures with a timeout
    argument). For those, we must not patch time.sleep or asyncio.sleep, or the
    timeout logic will not trigger correctly. We detect these by function name
    containing "timeout" and skip patching in that case.
    """

    def _sleep_noop(_secs: float = 0.0):
        return None

    async def _async_sleep_noop(_secs: float = 0.0):
        return None

    name = getattr(request.node, "name", "").lower()
    if "timeout" in name:
        # Do not patch sleeps for timeout-specific tests
        yield
    else:
        with (
            patch.object(_time, "sleep", _sleep_noop),
            patch.object(_asyncio, "sleep", _async_sleep_noop),
        ):
            yield


@pytest.fixture
def perf_counter_boundary():
    """Provide a deterministic perf_counter sequence for timing assertions."""
    seq = [0.0, 0.015, 0.030, 0.045, 0.060, 0.075]
    with patch("time.perf_counter", side_effect=seq):
        yield {"sequence": seq}


@pytest.fixture
def integration_settings():
    """Lightweight DocMindSettings for unit-tier integration checks."""
    return DocMindSettings(debug=False, log_level="INFO", enable_gpu_acceleration=False)


@pytest.fixture
def mock_llm() -> Mock:
    """Generic Mock LLM with simple complete/predict methods for unit tests."""
    llm = Mock()
    response = Mock()
    response.text = "Mock LLM response"
    llm.complete.return_value = response
    llm.invoke.return_value = "Mock LLM response"
    llm.predict.return_value = "Mock LLM response"
    return llm


# --- Unit-tier network and logging/system boundaries ---


@pytest.fixture(autouse=True)
def network_guard():
    """Guard against accidental network calls in unit tests (autouse)."""

    def _raise(*_a, **_k):
        raise RuntimeError("Network access disabled in unit tests")

    try:
        import httpx  # type: ignore
        import requests  # type: ignore

        with (
            patch.object(requests, "get", _raise),
            patch.object(requests, "post", _raise),
            patch.object(httpx, "get", _raise),
            patch.object(httpx, "post", _raise),
        ):
            yield
    except ImportError:
        # If libs are not present, do not patch
        yield


@pytest.fixture
def logging_boundary():
    """Structured logging boundary for src.utils.monitoring logger."""
    mock_logger = Mock()

    def assert_info_called(message_contains: str | None = None) -> None:
        mock_logger.info.assert_called()
        if message_contains is not None:
            call_msg = mock_logger.info.call_args[0][0]
            assert message_contains in call_msg

    def assert_error_called(message_contains: str | None = None) -> None:
        mock_logger.error.assert_called()
        if message_contains is not None:
            call_msg = mock_logger.error.call_args[0][0]
            assert message_contains in call_msg

    def assert_warning_called(message_contains: str | None = None) -> None:
        mock_logger.warning.assert_called()
        if message_contains is not None:
            call_msg = mock_logger.warning.call_args[0][0]
            assert message_contains in call_msg

    with patch("src.utils.monitoring.logger", mock_logger):
        yield {
            "logger": mock_logger,
            "assert_info_called": assert_info_called,
            "assert_error_called": assert_error_called,
            "assert_warning_called": assert_warning_called,
            "call_count": lambda level: getattr(mock_logger, level).call_count,
        }


@pytest.fixture
def performance_boundary():
    """Deterministic timing boundary using a fixed perf_counter sequence."""
    seq = [0.0, 0.015, 0.030, 0.045, 0.060, 0.075]
    with patch("time.perf_counter", side_effect=seq):
        yield {
            "timing_sequence": seq,
            "expected_step_ms": 15.0,
        }


@pytest.fixture
def system_resource_boundary():
    """Boundary for psutil and OS resource calls with predictable values."""
    import os as _os  # type: ignore
    from types import SimpleNamespace

    fake_proc = Mock()
    fake_proc.memory_info.return_value = SimpleNamespace(
        rss=100 * 1024 * 1024,
        vms=150 * 1024 * 1024,
    )
    fake_proc.memory_percent.return_value = 42.0

    with (
        patch("psutil.Process", return_value=fake_proc),
        patch("psutil.virtual_memory", return_value=SimpleNamespace(percent=65.0)),
        patch("psutil.cpu_percent", return_value=35.5),
        patch("psutil.disk_usage", return_value=SimpleNamespace(percent=45.0)),
        patch.object(_os, "getloadavg", return_value=(1.2, 1.5, 1.1), create=True),
    ):
        yield {
            "rss_mb": 100.0,
            "vms_mb": 150.0,
            "cpu_percent": 35.5,
            "disk_percent": 45.0,
            "loadavg": (1.2, 1.5, 1.1),
        }
