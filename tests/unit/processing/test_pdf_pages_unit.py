"""Unit test (skipped) for pHash metadata presence on PDF page images."""

import os

import pytest


@pytest.mark.skip(reason="requires PyMuPDF and sample PDF; covered in integration")
def test_pdf_pages_phash_present(tmp_path):
    """Ensure pHash field is present in image document metadata."""
    from src.processing.pdf_pages import pdf_pages_to_image_documents

    sample_pdf = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "test_data", "sample.pdf"
    )
    if not os.path.exists(sample_pdf):
        pytest.skip("sample.pdf not available")
    docs, _ = pdf_pages_to_image_documents(sample_pdf, dpi=200)
    assert docs
    assert "phash" in docs[0].metadata
