"""Policy tests for ``OcrController`` decision logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.processing import ProcessingStrategy
from src.processing.ocr_controller import OcrController, OcrDecision, OcrFeatures


@pytest.fixture
def controller() -> OcrController:
    return OcrController(pdf_sample_pages=2, large_pdf_page_threshold=5)


def _make_features(**overrides) -> OcrFeatures:
    base = OcrFeatures(
        file_path=Path("sample.pdf"),
        mime_type="application/pdf",
        extension=".pdf",
        size_bytes=1024,
        page_count=3,
        has_native_text=False,
    )
    data = {**base.__dict__, **overrides}
    return OcrFeatures(**data)


def test_text_extensions_short_circuit(controller: OcrController) -> None:
    features = _make_features(extension=".txt")
    decision = controller.decide(features)
    assert decision == OcrDecision(
        strategy=ProcessingStrategy.FAST,
        reason_code="EXTENSION_NATIVE",
        details={"extension": ".txt"},
    )


def test_image_extensions_short_circuit(controller: OcrController) -> None:
    features = _make_features(extension=".png")
    decision = controller.decide(features)
    assert decision.strategy is ProcessingStrategy.OCR_ONLY
    assert decision.reason_code == "IMAGE_FILE"


def test_pdf_native_text_short_circuit(
    controller: OcrController, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = _make_features(has_native_text=None)
    monkeypatch.setattr(controller, "_detect_pdf_native_text", lambda path: True)
    decision = controller.decide(features)
    assert decision.strategy is ProcessingStrategy.FAST
    assert decision.reason_code == "PDF_NATIVE_TEXT"


def test_pdf_scanned_hi_res_threshold(
    controller: OcrController, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = _make_features(has_native_text=False, page_count=None)
    monkeypatch.setattr(controller, "_detect_pdf_native_text", lambda path: False)
    monkeypatch.setattr(controller, "_count_pdf_pages", lambda path: 42)
    decision = controller.decide(features)
    assert decision.strategy is ProcessingStrategy.HI_RES
    assert decision.reason_code == "PDF_LARGE_SCANNED"


def test_pdf_scanned_default(
    controller: OcrController, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = _make_features(has_native_text=False, page_count=None)
    monkeypatch.setattr(controller, "_detect_pdf_native_text", lambda path: False)
    monkeypatch.setattr(controller, "_count_pdf_pages", lambda path: 2)
    decision = controller.decide(features)
    assert decision.strategy is ProcessingStrategy.OCR_ONLY
    assert decision.reason_code == "PDF_SCANNED"


def test_unknown_extension_defaults_to_fast(controller: OcrController) -> None:
    features = _make_features(extension=".zip")
    decision = controller.decide(features)
    assert decision.strategy is ProcessingStrategy.FAST
    assert decision.reason_code == "DEFAULT_FAST"
