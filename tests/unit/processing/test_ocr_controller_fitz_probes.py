"""Unit tests for OCR controller PDF probe helpers (fitz stubs)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.processing.ocr_controller import OcrController

pytestmark = pytest.mark.unit


def test_detect_pdf_native_text_probes_pages_via_fitz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitz = ModuleType("fitz")

    class _Page:
        def __init__(self, text: str) -> None:
            self._text = text

        def get_text(self):  # type: ignore[no-untyped-def]
            return self._text

    class _Doc:
        page_count = 3

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_a):  # type: ignore[no-untyped-def]
            return False

        def load_page(self, index: int):  # type: ignore[no-untyped-def]
            return _Page("hello" if index == 0 else "")

    fitz.open = lambda _p: _Doc()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fitz", fitz)

    c = OcrController(pdf_sample_pages=2)
    assert c._detect_pdf_native_text(Path("x.pdf")) is True


def test_detect_pdf_native_text_returns_false_when_no_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitz = ModuleType("fitz")

    class _Page:
        def get_text(self):  # type: ignore[no-untyped-def]
            return "   "

    class _Doc:
        page_count = 2

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_a):  # type: ignore[no-untyped-def]
            return False

        def load_page(self, _index: int):  # type: ignore[no-untyped-def]
            return _Page()

    fitz.open = lambda _p: _Doc()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fitz", fitz)

    c = OcrController(pdf_sample_pages=2)
    assert c._detect_pdf_native_text(Path("x.pdf")) is False


def test_count_pdf_pages_uses_fitz(monkeypatch: pytest.MonkeyPatch) -> None:
    fitz = ModuleType("fitz")

    class _Doc:
        page_count = 7

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_a):  # type: ignore[no-untyped-def]
            return False

    fitz.open = lambda _p: _Doc()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fitz", fitz)

    assert OcrController._count_pdf_pages(Path("x.pdf")) == 7
