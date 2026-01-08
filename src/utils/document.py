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
    raise NotImplementedError(
        f"{feature} is unavailable until the ingestion pipeline rewrite is complete."
    )


async def load_documents_unstructured(
    _file_paths: list[str | Path], _settings: Any | None = None
) -> list[Any]:
    """Placeholder coroutine for the upcoming ingestion helpers."""
    _raise_pending("load_documents_unstructured")


async def load_documents_from_directory(
    _directory_path: str | Path,
    _settings: Any | None = None,
    _recursive: bool = True,
    _supported_extensions: set[str] | None = None,
) -> list[Any]:
    """Placeholder coroutine for directory ingestion."""
    _raise_pending("load_documents_from_directory")


async def load_documents(_file_paths: list[str | Path]) -> list[Any]:
    """Convenience alias retained for compatibility with async callers."""
    _raise_pending("load_documents")


def ensure_spacy_model() -> Any:
    """Placeholder spaCy loader."""
    _raise_pending("ensure_spacy_model")


def get_document_info(file_path: str | Path) -> dict[str, Any]:
    """Return basic file metadata placeholder."""
    logger.debug("get_document_info pending ingestion refactor", file_path=file_path)
    _raise_pending("get_document_info")


def clear_document_cache() -> None:
    """Placeholder cache cleanup."""
    _raise_pending("clear_document_cache")


def get_cache_stats() -> dict[str, Any]:
    """Placeholder cache statistics."""
    _raise_pending("get_cache_stats")
