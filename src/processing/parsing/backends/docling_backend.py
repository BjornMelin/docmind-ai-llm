"""Docling document conversion adapter."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.processing.parsing.canonical_types import (
    DocumentParseResult,
    PageParseResult,
    ParserFramework,
    ParserProfile,
    ParserVersions,
)
from src.processing.parsing.errors import DocumentParseError
from src.processing.parsing.model_artifacts import (
    DOCLING_LAYOUT_BUNDLE,
    ModelIntegrityError,
    install_downloaded_model_bundle,
    model_files,
    verify_model_bundle,
)


def convert_with_docling(
    path: Path,
    *,
    document_id: str,
    source_hash: str,
    model_cache_dir: Path,
    max_pages: int,
    max_file_size: int,
    versions: ParserVersions,
) -> DocumentParseResult:
    """Convert a local document with Docling and return canonical output.

    Args:
        path: Local source document to convert.
        document_id: Stable document identifier for canonical page IDs.
        source_hash: SHA-256 digest of the source document.
        model_cache_dir: Application-owned cache containing pinned layout models.
        max_pages: Maximum number of physical pages to convert.
        max_file_size: Maximum accepted source size in bytes.
        versions: Parser package versions recorded in the result.

    Returns:
        DocumentParseResult: Canonical document and physical-page output.

    Raises:
        DocumentParseError: If Docling is unavailable or conversion is unsuccessful.
        ModelIntegrityError: If the pinned layout model bundle fails verification.
        RuntimeError: If required Docling converter components are unavailable.
    """
    try:
        from docling.datamodel.base_models import ConversionStatus
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise DocumentParseError(
            path,
            stage="docling_conversion",
            reason="docling_unavailable",
            cause=exc,
        ) from exc

    converter = _docling_converter(model_cache_dir=str(Path(model_cache_dir)))
    converted = converter.convert(
        Path(path),
        max_file_size=max_file_size,
        max_num_pages=max_pages,
    )
    status = getattr(converted, "status", None)
    if status is not ConversionStatus.SUCCESS:
        status_value = (
            status.value if isinstance(status, ConversionStatus) else "unknown"
        )
        raise DocumentParseError(
            path,
            stage="docling_conversion",
            reason=f"conversion_status_{status_value}",
        )
    document = getattr(converted, "document", converted)
    pages = _export_pages(document, document_id=document_id, max_pages=max_pages)
    return DocumentParseResult(
        document_id=document_id,
        source_filename=Path(path).name,
        source_hash=source_hash,
        profile=ParserProfile.CPU_SAFE,
        parser_framework=ParserFramework.DOCLING,
        page_count=len(pages),
        pages=pages,
        versions=versions,
    )


def docling_layout_model_files(model_cache_dir: Path) -> dict[str, Path]:
    """Return required Docling layout model files in the app cache.

    Args:
        model_cache_dir: Application-owned parser model cache.

    Returns:
        dict[str, Path]: Manifest-relative names mapped to local model paths.
    """
    return model_files(model_cache_dir, DOCLING_LAYOUT_BUNDLE)


def missing_docling_layout_models(model_cache_dir: Path) -> list[str]:
    """Return Docling layout artifacts missing from the app cache.

    Args:
        model_cache_dir: Application-owned parser model cache.

    Returns:
        list[str]: Manifest-relative names for missing artifacts.
    """
    return [
        name
        for name, path in docling_layout_model_files(model_cache_dir).items()
        if not path.exists()
    ]


def prefetch_docling_layout_models(
    model_cache_dir: Path, *, force: bool = False
) -> Path:
    """Download Docling layout artifacts into the app-owned cache.

    Args:
        model_cache_dir: Application-owned parser model cache.
        force: Download and replace the pinned bundle even when already valid.

    Returns:
        Path: Verified root directory for the installed layout bundle.

    Raises:
        RuntimeError: If Docling layout download support is unavailable.
        ModelIntegrityError: If the downloaded bundle fails verification.
    """
    try:
        from docling.models.stages.layout.layout_model import LayoutModel
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("docling is required to prefetch layout models") from exc
    cache_dir = Path(model_cache_dir)
    if not force:
        try:
            return verify_model_bundle(cache_dir, DOCLING_LAYOUT_BUNDLE)
        except ModelIntegrityError:
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".docling-layout-", dir=cache_dir) as temp_dir:
        download_dir = Path(temp_dir) / "download"
        LayoutModel.download_models(
            local_dir=download_dir,
            force=force,
            progress=True,
            layout_model_config=_pinned_docling_layout_config(),
        )
        return install_downloaded_model_bundle(
            download_dir,
            cache_dir,
            DOCLING_LAYOUT_BUNDLE,
        )


def verify_docling_layout_models(model_cache_dir: Path) -> Path:
    """Verify the exact pinned Docling layout bundle.

    Args:
        model_cache_dir: Application-owned parser model cache.

    Returns:
        Path: Verified root directory for the pinned layout bundle.

    Raises:
        ModelIntegrityError: If an expected artifact is missing or mismatched.
    """
    return verify_model_bundle(model_cache_dir, DOCLING_LAYOUT_BUNDLE)


def _pinned_docling_layout_config() -> Any:
    """Return Docling's native layout config with an immutable revision."""
    try:
        from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_HERON
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "docling is required for layout model configuration"
        ) from exc
    return DOCLING_LAYOUT_HERON.model_copy(
        update={
            "repo_id": DOCLING_LAYOUT_BUNDLE.repo_id,
            "revision": DOCLING_LAYOUT_BUNDLE.revision,
        }
    )


@lru_cache(maxsize=4)
def _docling_converter(*, model_cache_dir: str) -> Any:
    try:
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import LayoutOptions, PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("docling is required for document conversion") from exc

    verify_docling_layout_models(Path(model_cache_dir))
    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = False
    pdf_options.do_table_structure = False
    pdf_options.artifacts_path = str(Path(model_cache_dir))
    pdf_options.layout_options = LayoutOptions(
        model_spec=_pinned_docling_layout_config()
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )


def _export_pages(
    document: Any,
    *,
    document_id: str,
    max_pages: int,
) -> list[PageParseResult]:
    page_map = getattr(document, "pages", None)
    page_numbers = sorted(page_map) if isinstance(page_map, dict) else []
    if len(page_numbers) > max_pages:
        raise ValueError(f"document exceeds page limit of {max_pages}")
    if not page_numbers:
        text = _export_markdown(document)
        return [
            PageParseResult(
                page_id=f"{document_id}::page::1",
                page_index=0,
                text_markdown=text,
                routing_reason="docling_conversion",
            )
        ]
    return [
        PageParseResult(
            page_id=f"{document_id}::page::{page_index + 1}",
            page_index=page_index,
            text_markdown=_export_markdown(document, page_no=int(page_no)),
            routing_reason="docling_native_page",
        )
        for page_index, page_no in enumerate(page_numbers)
    ]


def _export_markdown(document: Any, *, page_no: int | None = None) -> str:
    for method in ("export_to_markdown", "export_to_text"):
        candidate = getattr(document, method, None)
        if callable(candidate):
            value = candidate(page_no=page_no) if page_no is not None else candidate()
            if value is not None:
                return str(value)
    return str(document or "")


__all__ = [
    "convert_with_docling",
    "docling_layout_model_files",
    "missing_docling_layout_models",
    "prefetch_docling_layout_models",
    "verify_docling_layout_models",
]
