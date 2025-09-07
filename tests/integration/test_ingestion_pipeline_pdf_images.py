"""Integration test (skipped) for PDF images ingestion with pHash and modality."""

import pytest


@pytest.mark.skip(reason="requires sample PDFs and full pipeline")
def test_ingestion_pdf_images_modality_and_phash():
    """Placeholder to assert modality=pdf_page_image and phash presence."""
    assert True  # implemented in full pipeline tests
