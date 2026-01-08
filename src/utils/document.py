"""Document processing utility placeholders.

Legacy ingestion helpers were removed as part of the 2025-09-16
ingestion/observability refactor. The functions exposed here now raise
``NotImplementedError`` so that any accidental usage surfaces immediately
until the new LlamaIndex-first pipeline lands in later phases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

from loguru import logger


def _raise_pending(feature: str) -> NoReturn:
    # TODO(ingestion-phase-2): Replace placeholders with the LlamaIndex-first
    # ingestion pipeline and remove this guard once implemented.
    raise NotImplementedError(
        f"{feature} is unavailable until the ingestion pipeline rewrite is complete."
    )


async def load_documents_unstructured(
    _file_paths: list[str | Path], _settings: Any | None = None
) -> list[Any]:
    """Placeholder coroutine for the upcoming ingestion helpers."""
    # TODO(ingestion-phase-2): Implement unstructured ingestion for file paths.
    _raise_pending("load_documents_unstructured")


async def load_documents_from_directory(
    _directory_path: str | Path,
    _settings: Any | None = None,
    _recursive: bool = True,
    _supported_extensions: set[str] | None = None,
) -> list[Any]:
    """Placeholder coroutine for directory ingestion."""
    # TODO(ingestion-phase-2): Implement directory ingestion with extension
    # filtering and recursive traversal.
    _raise_pending("load_documents_from_directory")


async def load_documents(_file_paths: list[str | Path]) -> list[Any]:
    """Convenience alias retained for compatibility with async callers."""
    # TODO(ingestion-phase-2): Route to the new ingestion pipeline.
    _raise_pending("load_documents")


def ensure_spacy_model() -> Any:
    """Placeholder spaCy loader."""
    # TODO(ingestion-phase-2): Restore spaCy model download/validation.
    _raise_pending("ensure_spacy_model")


def get_document_info(file_path: str | Path) -> dict[str, Any]:
    """Return basic file metadata placeholder."""
    # TODO(ingestion-phase-2): Populate document metadata via the new pipeline.
    logger.debug("get_document_info pending ingestion refactor", file_path=file_path)
    _raise_pending("get_document_info")


def clear_document_cache() -> None:
    """Placeholder cache cleanup."""
    # TODO(ingestion-phase-2): Implement cache invalidation for ingested docs.
    _raise_pending("clear_document_cache")


def get_cache_stats() -> dict[str, Any]:
    """Placeholder cache statistics."""
    # TODO(ingestion-phase-2): Provide cache statistics for ingestion artifacts.
    _raise_pending("get_cache_stats")
