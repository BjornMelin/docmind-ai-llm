"""Simplified document loading utilities for DocMind AI.

This module provides core offline document processing using LlamaIndex UnstructuredReader
with high-resolution parsing strategy for multimodal content extraction.

ADR-004 Compliance: Offline document loading with Unstructured hi_res strategy.

Supported formats:
- Documents: PDF, DOCX, PPTX, HTML, TXT, MD
- Multimodal: Text, images, tables with OCR via YOLOX/Tesseract

Key features:
- UnstructuredReader with hi_res strategy (ADR-004)
- Simple diskcache for document caching
- Basic error handling with loguru
- Multimodal support for text, images, tables

Example:
    Basic document loading::

        from utils.document_loader import load_documents_unstructured

        # Load with multimodal support
        docs = load_documents_unstructured("document.pdf")

        # Check loaded documents
        for doc in docs:
            print(f"Loaded: {doc.metadata.get('filename')}")
"""

import hashlib
from pathlib import Path
from typing import Any

import diskcache
from llama_index.core import Document
from llama_index.readers.file import UnstructuredReader
from loguru import logger

from models.core import settings
from utils.exceptions import DocumentLoadingError, handle_document_error
from utils.retry_utils import document_retry

# Simple document cache using diskcache
_document_cache = diskcache.Cache("./cache/documents")


def _get_file_hash(file_path: str) -> str:
    """Generate SHA-256 hash for file content."""
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()
    except Exception as e:
        logger.warning(f"Could not hash file {file_path}: {e}")
        return f"nohash_{Path(file_path).name}_{Path(file_path).stat().st_mtime}"


def _is_cached(file_path: str) -> bool:
    """Check if document is cached."""
    file_hash = _get_file_hash(file_path)
    return file_hash in _document_cache


def _get_cached(file_path: str) -> list[Document] | None:
    """Get cached documents."""
    if not _is_cached(file_path):
        return None
    file_hash = _get_file_hash(file_path)
    return _document_cache.get(file_hash)


def _cache_documents(file_path: str, documents: list[Document]) -> None:
    """Cache documents with 1 hour expiry."""
    file_hash = _get_file_hash(file_path)
    _document_cache.set(file_hash, documents, expire=3600)


@document_retry
def load_documents_unstructured(file_path: str | Path) -> list[Document]:
    """Load documents using UnstructuredReader with hi_res strategy (ADR-004).

    Uses Unstructured library for offline multimodal document processing including
    text, images, and tables with OCR support via YOLOX/Tesseract.

    Args:
        file_path: Path to document file to process

    Returns:
        List of Document objects with multimodal content and metadata

    Raises:
        DocumentLoadingError: If document loading fails after retries

    Note:
        Follows ADR-004 specification for offline document parsing using
        UnstructuredReader.load_data(file_path, strategy='hi_res')
    """
    file_path = str(file_path)

    # Check cache first
    cached_docs = _get_cached(file_path)
    if cached_docs:
        logger.info(f"Using cached document for {Path(file_path).name}")
        return cached_docs

    logger.info(f"Loading document with UnstructuredReader: {Path(file_path).name}")

    try:
        # Validate file exists
        if not Path(file_path).exists():
            raise DocumentLoadingError(f"File not found: {file_path}")

        # Initialize UnstructuredReader (ADR-004 compliant)
        reader = UnstructuredReader()

        # Load with hi_res strategy for multimodal extraction (ADR-004)
        documents = reader.load_data(
            file=file_path,
            strategy=getattr(settings, "parse_strategy", "hi_res"),
            split_documents=True,
        )

        # Enhance metadata
        for doc in documents:
            if not hasattr(doc, "metadata"):
                doc.metadata = {}
            doc.metadata.update(
                {
                    "filename": Path(file_path).name,
                    "source": str(file_path),
                    "loader": "unstructured_reader",
                    "strategy": getattr(settings, "parse_strategy", "hi_res"),
                }
            )

        # Cache results
        _cache_documents(file_path, documents)

        logger.success(f"Loaded {len(documents)} elements from {Path(file_path).name}")
        return documents

    except Exception as e:
        error_msg = f"Failed to load document {Path(file_path).name}: {str(e)}"
        logger.error(error_msg)

        raise handle_document_error(
            e, operation="unstructured_document_loading", file_path=file_path
        ) from e


def load_documents_from_directory(directory_path: str | Path) -> list[Document]:
    """Load all supported documents from a directory.

    Args:
        directory_path: Path to directory containing documents

    Returns:
        List of all loaded Document objects

    Raises:
        DocumentLoadingError: If directory access or document loading fails
    """
    directory_path = Path(directory_path)

    if not directory_path.exists() or not directory_path.is_dir():
        raise DocumentLoadingError(f"Directory not found: {directory_path}")

    # Supported file extensions
    supported_extensions = {".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md"}

    # Find all supported files
    files = []
    for ext in supported_extensions:
        files.extend(directory_path.glob(f"*{ext}"))
        files.extend(directory_path.glob(f"**/*{ext}"))  # Recursive search

    if not files:
        logger.warning(f"No supported documents found in {directory_path}")
        return []

    logger.info(f"Found {len(files)} documents in {directory_path}")

    # Load all documents
    all_documents = []
    for file_path in files:
        try:
            documents = load_documents_unstructured(file_path)
            all_documents.extend(documents)
        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {e}")
            continue

    logger.success(
        f"Loaded {len(all_documents)} total elements from {len(files)} files"
    )
    return all_documents


def get_document_info(file_path: str | Path) -> dict[str, Any]:
    """Get basic information about a document without loading it.

    Args:
        file_path: Path to document file

    Returns:
        Dictionary with document information (size, type, cached status)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return {"error": "File not found"}

    stat = file_path.stat()

    return {
        "filename": file_path.name,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / 1024 / 1024, 2),
        "extension": file_path.suffix.lower(),
        "modified": stat.st_mtime,
        "is_cached": _is_cached(str(file_path)),
    }


def clear_document_cache() -> int:
    """Clear all cached documents.

    Returns:
        Number of cached items cleared
    """
    try:
        count = len(_document_cache)
        _document_cache.clear()
        logger.info(f"Cleared {count} cached documents")
        return count
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return 0


def get_cache_stats() -> dict[str, Any]:
    """Get document cache statistics.

    Returns:
        Dictionary with cache statistics
    """
    try:
        return {
            "cache_size": len(_document_cache),
            "cache_dir": str(_document_cache.directory),
            "total_size_bytes": sum(
                Path(_document_cache.directory).glob("**/*").st_size
                for p in Path(_document_cache.directory).glob("**/*")
                if p.is_file()
            )
            if Path(_document_cache.directory).exists()
            else 0,
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {"error": str(e)}


# Backwards compatibility aliases for existing code
load_documents_llama = load_documents_unstructured  # Legacy alias
