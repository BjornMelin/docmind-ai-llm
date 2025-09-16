"""Unit tests for the OCR controller policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.processing import ProcessingStrategy
from src.processing.ocr_controller import OcrController, OcrFeatures


@pytest.fixture
def controller() -> OcrController:
    return OcrController(pdf_sample_pages=1, large_pdf_page_threshold=10)


def test_text_extension_returns_fast(controller: OcrController, tmp_path: Path) -> None:
    doc = tmp_path / "sample.txt"
    doc.write_text("hello")
    features = OcrFeatures(
        file_path=doc,
        mime_type="text/plain",
        extension=".txt",
        size_bytes=doc.stat().st_size,
    )
    decision = controller.decide(features)
    assert decision.strategy == ProcessingStrategy.FAST
    assert decision.reason_code == "EXTENSION_NATIVE"


def test_image_extension_returns_ocr_only(
    controller: OcrController, tmp_path: Path
) -> None:
    img = tmp_path / "photo.png"
    img.write_bytes(b"fakepng")
    features = OcrFeatures(
        file_path=img,
        mime_type="image/png",
        extension=".png",
        size_bytes=img.stat().st_size,
    )
    decision = controller.decide(features)
    assert decision.strategy == ProcessingStrategy.OCR_ONLY
    assert decision.reason_code == "IMAGE_FILE"


def test_pdf_native_text_shortcuts_to_fast(
    monkeypatch, controller: OcrController, tmp_path: Path
) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")

    def _fake_detect(_: Path) -> bool:
        return True

    monkeypatch.setattr(controller, "_detect_pdf_native_text", _fake_detect)
    features = OcrFeatures(
        file_path=pdf,
        mime_type="application/pdf",
        extension=".pdf",
        size_bytes=0,
    )
    decision = controller.decide(features)
    assert decision.strategy == ProcessingStrategy.FAST
    assert decision.reason_code == "PDF_NATIVE_TEXT"


def test_pdf_large_scanned_prefers_hi_res(
    monkeypatch, controller: OcrController, tmp_path: Path
) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")

    monkeypatch.setattr(controller, "_detect_pdf_native_text", lambda _: False)
    monkeypatch.setattr(controller, "_count_pdf_pages", lambda _path: 25)

    features = OcrFeatures(
        file_path=pdf,
        mime_type="application/pdf",
        extension=".pdf",
        size_bytes=0,
    )
    decision = controller.decide(features)
    assert decision.strategy == ProcessingStrategy.HI_RES
    assert decision.reason_code == "PDF_LARGE_SCANNED"
