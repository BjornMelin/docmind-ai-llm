"""pypdfium2-backed PDF inspection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from src.processing.parsing.errors import DocumentParseError
from src.utils.log_safety import build_pii_log_entry


@dataclass(frozen=True, slots=True)
class PdfPageInspection:
    """Lightweight inspection result for one PDF page."""

    page_index: int
    width: float
    height: float
    text: str

    @property
    def has_text(self) -> bool:
        """Return whether this page has meaningful native text."""
        return bool(self.text.strip())


@dataclass(frozen=True, slots=True)
class PdfInspectionResult:
    """Lightweight inspection result for a PDF."""

    page_count: int
    pages: list[PdfPageInspection]

    @property
    def has_native_text(self) -> bool:
        """Return whether any sampled page has meaningful native text."""
        return any(page.has_text for page in self.pages)


def inspect_pdf(
    pdf_path: Path,
    *,
    sample_pages: int = 3,
    max_text_chars: int = 8000,
    max_pages: int | None = None,
    render_dpi: int = 200,
    max_render_pixels: int | None = None,
) -> PdfInspectionResult:
    """Inspect PDF page count and native text with pypdfium2.

    Args:
        pdf_path: Local PDF path.
        sample_pages: Number of leading pages to sample for text.
        max_text_chars: Maximum text chars retained per sampled page.
        max_pages: Optional hard page-count limit.
        render_dpi: Planned render resolution used for pixel-limit validation.
        max_render_pixels: Optional hard pixel limit for any rendered page.

    Returns:
        PdfInspectionResult with page count and sampled page text.

    Raises:
        ValueError: If an inspection limit is invalid.
        DocumentParseError: If pypdfium2 is unavailable, the PDF is malformed,
            or a document resource limit is exceeded.
    """
    if sample_pages < 0:
        raise ValueError("sample_pages must be non-negative")
    if max_text_chars < 0:
        raise ValueError("max_text_chars must be non-negative")
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive when provided")
    if render_dpi <= 0:
        raise ValueError("render_dpi must be positive")
    if max_render_pixels is not None and max_render_pixels <= 0:
        raise ValueError("max_render_pixels must be positive when provided")

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise DocumentParseError(
            pdf_path,
            stage="pdf_inspection",
            reason="pypdfium2_unavailable",
            cause=exc,
        ) from exc

    path = Path(pdf_path)
    pages: list[PdfPageInspection] = []
    try:
        with pdfium.PdfDocument(path) as doc:
            page_count = len(doc)
            if max_pages is not None and page_count > max_pages:
                raise DocumentParseError(
                    path,
                    stage="pdf_inspection",
                    reason="page_limit_exceeded",
                )
            for index in range(min(page_count, sample_pages)):
                page = doc[index]
                try:
                    textpage = page.get_textpage()
                    try:
                        text = str(textpage.get_text_range() or "").strip()
                    finally:
                        textpage.close()
                    if len(text) > max_text_chars:
                        text = text[:max_text_chars]
                    width, height = _page_size(page)
                    if max_render_pixels is not None:
                        scale = render_dpi / 72.0
                        pixels = int(width * scale) * int(height * scale)
                        if pixels > max_render_pixels:
                            raise DocumentParseError(
                                path,
                                stage="pdf_inspection",
                                reason="render_pixel_limit_exceeded",
                            )
                    pages.append(
                        PdfPageInspection(
                            page_index=index,
                            width=width,
                            height=height,
                            text=text,
                        )
                    )
                finally:
                    page.close()
    except DocumentParseError:
        raise
    except (RuntimeError, ValueError, OSError) as exc:
        redaction = build_pii_log_entry(str(exc), key_id="pdf_inspector")
        logger.debug(
            "PDF inspection failed (file={}, error_type={}, error={})",
            path.name,
            type(exc).__name__,
            redaction.redacted,
        )
        raise DocumentParseError(
            path,
            stage="pdf_inspection",
            reason="invalid_or_unreadable_pdf",
            cause=exc,
        ) from exc
    return PdfInspectionResult(page_count=page_count, pages=pages)


def _page_size(page: Any) -> tuple[float, float]:
    try:
        width, height = page.get_size()
        return float(width), float(height)
    except (AttributeError, TypeError, ValueError):
        return float(page.get_width()), float(page.get_height())


__all__ = ["PdfInspectionResult", "PdfPageInspection", "inspect_pdf"]
