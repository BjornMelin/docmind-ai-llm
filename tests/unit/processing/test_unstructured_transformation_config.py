"""Unit tests for UnstructuredTransformation partition configuration.

Validates kwargs produced for Unstructured partition() under each strategy.
"""

from __future__ import annotations

import pytest

from src.models.processing import ProcessingStrategy
from src.processing.document_processor import UnstructuredTransformation


@pytest.mark.unit
def test_partition_config_hi_res() -> None:
    """HI_RES should enable image extraction and table inference."""
    tfm = UnstructuredTransformation(strategy=ProcessingStrategy.HI_RES)
    cfg = tfm._build_partition_config(ProcessingStrategy.HI_RES)
    assert cfg["strategy"] == ProcessingStrategy.HI_RES.value
    assert cfg["include_metadata"] is True
    assert cfg["include_page_breaks"] is True
    assert cfg["extract_images_in_pdf"] is True
    assert cfg["extract_image_blocks"] is True
    assert cfg["infer_table_structure"] is True


@pytest.mark.unit
def test_partition_config_fast() -> None:
    """FAST should disable image/table heavy features."""
    tfm = UnstructuredTransformation(strategy=ProcessingStrategy.FAST)
    cfg = tfm._build_partition_config(ProcessingStrategy.FAST)
    assert cfg["strategy"] == ProcessingStrategy.FAST.value
    assert cfg["include_metadata"] is True
    assert cfg["include_page_breaks"] is True
    assert cfg["extract_images_in_pdf"] is False
    assert cfg["infer_table_structure"] is False


@pytest.mark.unit
def test_partition_config_ocr_only() -> None:
    """OCR_ONLY should enable OCR-related flags and languages."""
    tfm = UnstructuredTransformation(strategy=ProcessingStrategy.OCR_ONLY)
    cfg = tfm._build_partition_config(ProcessingStrategy.OCR_ONLY)
    assert cfg["strategy"] == ProcessingStrategy.OCR_ONLY.value
    assert cfg["include_metadata"] is True
    assert cfg["include_page_breaks"] is True
    assert cfg["extract_images_in_pdf"] is True
    assert cfg["extract_image_blocks"] is True
    assert cfg["infer_table_structure"] is False
    assert cfg["ocr_languages"] == ["eng"]
