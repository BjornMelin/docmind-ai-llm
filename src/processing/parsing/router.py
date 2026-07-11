"""Parser routing policy for local document ingestion."""

from __future__ import annotations

from pathlib import Path

from src.processing.parsing.canonical_types import ParserFramework
from src.processing.parsing.formats import DIRECT_TEXT_EXTENSIONS
from src.processing.parsing.pdf_inspector import PdfInspectionResult


def choose_framework(path: Path) -> ParserFramework:
    """Choose the parser framework for a source path."""
    suffix = path.suffix.lower()
    if suffix in DIRECT_TEXT_EXTENSIONS:
        return ParserFramework.DIRECT_TEXT
    return ParserFramework.DOCLING


def routing_reason(
    path: Path,
    *,
    inspection: PdfInspectionResult | None,
    force_ocr: bool,
) -> str:
    """Return a stable routing reason for parser provenance."""
    suffix = path.suffix.lower()
    if suffix in DIRECT_TEXT_EXTENSIONS:
        return "direct_text"
    if suffix == ".pdf" and inspection is not None:
        if force_ocr:
            return "pdf_force_ocr"
        if inspection.has_native_text:
            return "pdf_native_text"
        return "pdf_low_text_ocr_candidate"
    return "docling_multiformat"


__all__ = [
    "ParserFramework",
    "choose_framework",
    "routing_reason",
]
