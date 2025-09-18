"""Tests for the OCR controller rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.processing import ProcessingStrategy
from src.processing.ocr_controller import OcrController, OcrFeatures


@pytest.fixture
def controller() -> OcrController:
    """Return a controller with a small threshold for unit tests."""
    return OcrController(pdf_sample_pages=2, large_pdf_page_threshold=5)


def test_decide_handles_text_extensions(controller: OcrController) -> None:
    features = OcrFeatures(
        file_path=Path("doc.txt"),
        mime_type="text/plain",
        extension=".txt",
        size_bytes=128,
    )

    decision = controller.decide(features)
    assert decision.strategy is ProcessingStrategy.FAST
    assert decision.reason_code == "EXTENSION_NATIVE"


def test_decide_handles_image_extensions(controller: OcrController) -> None:
    features = OcrFeatures(
        file_path=Path("scan.png"),
        mime_type="image/png",
        extension=".png",
        size_bytes=2048,
    )

    decision = controller.decide(features)
    assert decision.strategy is ProcessingStrategy.OCR_ONLY
    assert decision.reason_code == "IMAGE_FILE"


def test_decide_pdf_with_native_text(
    controller: OcrController, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = OcrFeatures(
        file_path=Path("doc.pdf"),
        mime_type="application/pdf",
        extension=".pdf",
        size_bytes=1024,
        has_native_text=None,
    )

    monkeypatch.setattr(controller, "_detect_pdf_native_text", lambda _path: True)

    decision = controller.decide(features)
    assert decision.strategy is ProcessingStrategy.FAST
    assert decision.reason_code == "PDF_NATIVE_TEXT"


def test_decide_pdf_scanned_high_page_count(
    controller: OcrController, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = OcrFeatures(
        file_path=Path("scan.pdf"),
        mime_type="application/pdf",
        extension=".pdf",
        size_bytes=4096,
        has_native_text=False,
        page_count=None,
    )

    monkeypatch.setattr(controller, "_count_pdf_pages", lambda _path: 12)

    decision = controller.decide(features)
    assert decision.strategy is ProcessingStrategy.HI_RES
    assert decision.reason_code == "PDF_LARGE_SCANNED"


def test_decide_pdf_scanned_small(
    controller: OcrController, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = OcrFeatures(
        file_path=Path("scan.pdf"),
        mime_type="application/pdf",
        extension=".pdf",
        size_bytes=4096,
        has_native_text=False,
        page_count=2,
    )

    decision = controller.decide(features)
    assert decision.strategy is ProcessingStrategy.OCR_ONLY
    assert decision.reason_code == "PDF_SCANNED"


def test_decide_default_fast_for_unknown_types(controller: OcrController) -> None:
    features = OcrFeatures(
        file_path=Path("archive.zip"),
        mime_type="application/zip",
        extension=".zip",
        size_bytes=10_000,
    )

    decision = controller.decide(features)
    assert decision.strategy is ProcessingStrategy.FAST
    assert decision.reason_code == "DEFAULT_FAST"
