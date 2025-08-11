"""Tests for PDF document processing functionality.

This module tests PDF-specific functionality including image extraction
with modern pytest fixtures and proper typing.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.document_loader import extract_images_from_pdf

# Shared fixtures are automatically available via conftest.py


class TestExtractImagesFromPDF:
    """Test PDF image extraction functionality with modern fixtures."""

    @pytest.mark.parametrize(
        "page_count,expected_calls",
        [
            (1, 1),
            (2, 2),
            (5, 5),
        ],
    )
    @patch("utils.document_loader.fitz")
    def test_extract_images_success(
        self,
        mock_fitz: Mock,
        sample_pdf_path: str,
        mock_fitz_document: Mock,
        mock_fitz_page: Mock,
        mock_fitz_pixmap: Mock,
        page_count: int,
        expected_calls: int,
    ) -> None:
        """Test successful image extraction from PDF with varying page counts."""
        # Setup mocks
        mock_fitz_document.__len__.return_value = page_count
        mock_fitz_document.load_page.return_value = mock_fitz_page
        mock_fitz_page.get_images.return_value = [(123, "test", "image")]

        # Mock RGB pixmap
        mock_fitz_pixmap.n = 3  # RGB
        mock_fitz_pixmap.alpha = 0
        mock_fitz_pixmap.tobytes.return_value = b"mock ppm data"

        mock_fitz.open.return_value = mock_fitz_document
        mock_fitz.Pixmap.return_value = mock_fitz_pixmap

        # Mock PIL Image
        with patch("utils.document_loader.Image") as mock_pil:
            mock_img = Mock()
            mock_img.size = (100, 100)
            mock_pil.open.return_value = mock_img

            # Mock base64 encoding
            with patch("utils.document_loader.base64.b64encode") as mock_b64:
                mock_b64.return_value.decode.return_value = "mock_base64_data"

                with patch("utils.document_loader.logging.info") as mock_log_info:
                    result = extract_images_from_pdf(sample_pdf_path)

                    # Verify document operations
                    mock_fitz.open.assert_called_once_with(sample_pdf_path)
                    assert mock_fitz_document.load_page.call_count == expected_calls
                    mock_fitz_document.close.assert_called_once()

                    # Verify result structure
                    assert len(result) == page_count
                    for img_data in result:
                        assert img_data["image_data"] == "mock_base64_data"
                        assert img_data["format"] == "PNG"
                        assert img_data["size"] == (100, 100)
                        assert "page_number" in img_data
                        assert "image_index" in img_data

                    # Verify logging
                    mock_log_info.assert_called_with(
                        f"Extracted {page_count} images from PDF"
                    )

    @patch("utils.document_loader.fitz")
    def test_extract_images_no_images(
        self,
        mock_fitz: Mock,
        sample_pdf_path: str,
        mock_fitz_document: Mock,
        mock_fitz_page: Mock,
    ) -> None:
        """Test PDF image extraction when no images are found."""
        mock_fitz_document.__len__.return_value = 1
        mock_fitz_document.load_page.return_value = mock_fitz_page
        mock_fitz_page.get_images.return_value = []  # No images

        mock_fitz.open.return_value = mock_fitz_document

        with patch("utils.document_loader.logging.info") as mock_log_info:
            result = extract_images_from_pdf(sample_pdf_path)

            assert result == []
            mock_log_info.assert_called_with("Extracted 0 images from PDF")

    @pytest.mark.parametrize(
        "error_type,error_message",
        [
            (Exception("PyMuPDF error"), "PyMuPDF error"),
            (FileNotFoundError("File not found"), "File not found"),
            (MemoryError("Out of memory"), "Out of memory"),
        ],
    )
    @patch("utils.document_loader.fitz")
    def test_extract_images_fitz_errors(
        self,
        mock_fitz: Mock,
        sample_pdf_path: str,
        error_type: Exception,
        error_message: str,
    ) -> None:
        """Test PDF image extraction with various PyMuPDF errors."""
        mock_fitz.open.side_effect = error_type

        with patch("utils.document_loader.logging.error") as mock_log_error:
            result = extract_images_from_pdf(sample_pdf_path)

            assert result == []
            mock_log_error.assert_called_once()
            logged_message = mock_log_error.call_args[0][0]
            assert "PDF image extraction failed" in logged_message

    @pytest.mark.parametrize(
        "pixmap_n,pixmap_alpha,should_skip",
        [
            (3, 0, False),  # RGB, no alpha - should process
            (4, 1, False),  # RGB + alpha - should process
            (5, 1, True),  # CMYK + alpha - should skip
            (4, 0, True),  # CMYK, no alpha - should skip
        ],
    )
    @patch("utils.document_loader.fitz")
    def test_extract_images_pixmap_format_handling(
        self,
        mock_fitz: Mock,
        sample_pdf_path: str,
        mock_fitz_document: Mock,
        mock_fitz_page: Mock,
        mock_fitz_pixmap: Mock,
        pixmap_n: int,
        pixmap_alpha: int,
        should_skip: bool,
    ) -> None:
        """Test handling of different pixmap formats (RGB vs CMYK)."""
        mock_fitz_document.__len__.return_value = 1
        mock_fitz_document.load_page.return_value = mock_fitz_page
        mock_fitz_page.get_images.return_value = [(123, "test", "image")]

        # Configure pixmap format
        mock_fitz_pixmap.n = pixmap_n
        mock_fitz_pixmap.alpha = pixmap_alpha

        mock_fitz.open.return_value = mock_fitz_document
        mock_fitz.Pixmap.return_value = mock_fitz_pixmap

        if not should_skip:
            mock_fitz_pixmap.tobytes.return_value = b"mock ppm data"

            with (
                patch("utils.document_loader.Image") as mock_pil,
                patch("utils.document_loader.base64.b64encode") as mock_b64,
            ):
                mock_img = Mock()
                mock_img.size = (100, 100)
                mock_pil.open.return_value = mock_img
                mock_b64.return_value.decode.return_value = "mock_base64_data"

                with patch("utils.document_loader.logging.info") as mock_log_info:
                    result = extract_images_from_pdf(sample_pdf_path)

                    assert len(result) == 1
                    mock_log_info.assert_called_with("Extracted 1 images from PDF")
        else:
            with patch("utils.document_loader.logging.info") as mock_log_info:
                result = extract_images_from_pdf(sample_pdf_path)

                assert result == []
                mock_log_info.assert_called_with("Extracted 0 images from PDF")

    @patch("utils.document_loader.fitz")
    def test_extract_images_pil_error_handling(
        self,
        mock_fitz: Mock,
        sample_pdf_path: str,
        mock_fitz_document: Mock,
        mock_fitz_page: Mock,
        mock_fitz_pixmap: Mock,
    ) -> None:
        """Test handling of PIL Image processing errors."""
        mock_fitz_document.__len__.return_value = 1
        mock_fitz_document.load_page.return_value = mock_fitz_page
        mock_fitz_page.get_images.return_value = [(123, "test", "image")]

        mock_fitz_pixmap.n = 3
        mock_fitz_pixmap.alpha = 0
        mock_fitz_pixmap.tobytes.return_value = b"invalid ppm data"

        mock_fitz.open.return_value = mock_fitz_document
        mock_fitz.Pixmap.return_value = mock_fitz_pixmap

        # Mock PIL Image to raise an error
        with patch("utils.document_loader.Image") as mock_pil:
            mock_pil.open.side_effect = Exception("PIL processing error")

            with (
                patch("utils.document_loader.logging.warning") as mock_log_warning,
                patch("utils.document_loader.logging.info") as mock_log_info,
            ):
                result = extract_images_from_pdf(sample_pdf_path)

                # Should continue processing despite PIL error
                assert result == []
                mock_log_info.assert_called_with("Extracted 0 images from PDF")

                # Should log warning about the PIL error
                mock_log_warning.assert_called()
                warning_message = mock_log_warning.call_args[0][0]
                assert "Failed to process image" in warning_message

    @patch("utils.document_loader.fitz")
    def test_extract_images_memory_cleanup(
        self,
        mock_fitz: Mock,
        sample_pdf_path: str,
        mock_fitz_document: Mock,
        mock_fitz_page: Mock,
        mock_fitz_pixmap: Mock,
    ) -> None:
        """Test proper memory cleanup in case of errors."""
        mock_fitz_document.__len__.return_value = 1
        mock_fitz_document.load_page.return_value = mock_fitz_page
        mock_fitz_page.get_images.return_value = [(123, "test", "image")]

        mock_fitz_pixmap.n = 3
        mock_fitz_pixmap.alpha = 0
        mock_fitz_pixmap.tobytes.side_effect = Exception("Memory error during tobytes")

        mock_fitz.open.return_value = mock_fitz_document
        mock_fitz.Pixmap.return_value = mock_fitz_pixmap

        with patch("utils.document_loader.logging.error") as mock_log_error:
            result = extract_images_from_pdf(sample_pdf_path)

            # Should still close document despite error
            mock_fitz_document.close.assert_called_once()
            assert result == []

            # Should not log an error (handled gracefully)
            # Error is caught and handled within the page loop
