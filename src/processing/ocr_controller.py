"""OCR decision controller for ingestion pipeline.

The controller evaluates lightweight heuristics to determine which processing
strategy to apply to a document. It intentionally avoids external services and
relies on inexpensive file probes so we can keep the stack offline-friendly and
simple.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from loguru import logger

from src.models.processing import ProcessingStrategy
from src.utils.log_safety import build_pii_log_entry

_TEXT_EXTENSIONS = {".txt", ".md", ".rtf", ".html", ".htm"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}


@dataclass
class OcrFeatures:
    """Derived attributes describing a candidate document."""

    file_path: Path
    mime_type: str | None
    extension: str
    size_bytes: int
    page_count: int | None = None
    has_native_text: bool | None = None


@dataclass
class OcrDecision:
    """Outcome of an OCR policy evaluation."""

    strategy: ProcessingStrategy
    reason_code: str
    details: dict[str, object]


class OcrController:
    """Apply deterministic OCR policy rules."""

    def __init__(
        self,
        *,
        pdf_sample_pages: int = 3,
        large_pdf_page_threshold: int = 50,
    ) -> None:
        """Initialize the controller.

        Args:
            pdf_sample_pages: Number of pages to sample when probing PDFs for
                embedded text.
            large_pdf_page_threshold: Page count at which hi-res processing is
                preferred for scanned PDFs.
        """
        self._pdf_sample_pages = pdf_sample_pages
        self._large_pdf_page_threshold = large_pdf_page_threshold

    def decide(self, features: OcrFeatures) -> OcrDecision:
        """Choose an OCR strategy given document features.

        Args:
            features: Metadata describing the document.

        Returns:
            OcrDecision: Strategy selection with reason metadata.
        """
        ext = features.extension.lower()

        if ext in _TEXT_EXTENSIONS:
            return OcrDecision(
                strategy=ProcessingStrategy.FAST,
                reason_code="EXTENSION_NATIVE",
                details={"extension": ext},
            )

        if ext in _IMAGE_EXTENSIONS:
            return OcrDecision(
                strategy=ProcessingStrategy.OCR_ONLY,
                reason_code="IMAGE_FILE",
                details={"extension": ext},
            )

        if ext == ".pdf":
            has_text = features.has_native_text
            if has_text is None:
                has_text = self._detect_pdf_native_text(features.file_path)
            if has_text:
                return OcrDecision(
                    strategy=ProcessingStrategy.FAST,
                    reason_code="PDF_NATIVE_TEXT",
                    details={"sampled_pages": self._pdf_sample_pages},
                )

            pages = features.page_count
            if pages is None:
                pages = self._count_pdf_pages(features.file_path)
            if pages is not None and pages > self._large_pdf_page_threshold:
                return OcrDecision(
                    strategy=ProcessingStrategy.HI_RES,
                    reason_code="PDF_LARGE_SCANNED",
                    details={"page_count": pages},
                )

            return OcrDecision(
                strategy=ProcessingStrategy.OCR_ONLY,
                reason_code="PDF_SCANNED",
                details={"page_count": pages},
            )

        # Default: treat unknown types as FAST unless explicitly image-based.
        return OcrDecision(
            strategy=ProcessingStrategy.FAST,
            reason_code="DEFAULT_FAST",
            details={"extension": ext},
        )

    def _detect_pdf_native_text(self, file_path: Path) -> bool:
        """Probe a PDF for embedded text by sampling pages via PyMuPDF.

        Args:
            file_path: PDF path under evaluation.

        Returns:
            bool: ``True`` when any sampled page contains extractable text.
        """
        try:
            import fitz  # type: ignore

            with fitz.open(file_path) as doc:
                sample_count = min(doc.page_count, self._pdf_sample_pages)
                for index in range(sample_count):
                    page = doc.load_page(index)
                    text = cast(Any, page).get_text().strip()
                    if text:
                        return True
        except (
            ImportError,
            RuntimeError,
            ValueError,
            OSError,
        ) as exc:  # pragma: no cover - fallback path
            redaction = build_pii_log_entry(str(exc), key_id="ocr.native_text_probe")
            logger.debug(
                "native text probe failed (file={}, error_type={}, error={})",
                file_path.name,
                type(exc).__name__,
                redaction.redacted,
            )
        return False

    @staticmethod
    def _count_pdf_pages(file_path: Path) -> int | None:
        """Determine the number of pages in the provided PDF file.

        Args:
            file_path: PDF path under evaluation.

        Returns:
            int | None: Page count when available; otherwise ``None``.
        """
        try:
            import fitz  # type: ignore

            with fitz.open(file_path) as doc:
                return doc.page_count
        except (
            ImportError,
            RuntimeError,
            ValueError,
            OSError,
        ) as exc:  # pragma: no cover - fallback path
            redaction = build_pii_log_entry(str(exc), key_id="ocr.page_count_probe")
            logger.debug(
                "page count probe failed (file={}, error_type={}, error={})",
                file_path.name,
                type(exc).__name__,
                redaction.redacted,
            )
            return None
