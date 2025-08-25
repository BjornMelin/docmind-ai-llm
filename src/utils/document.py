"""Document loading and processing utilities for DocMind AI.

This module provides core document processing using LlamaIndex UnstructuredReader
with high-resolution parsing strategy for multimodal content extraction, plus
essential spaCy model management for NLP processing.

ADR-004 Compliance: Offline document loading with Unstructured hi_res strategy.

Supported formats:
- Documents: PDF, DOCX, PPTX, HTML, TXT, MD
- Multimodal: Text, images, tables with OCR via YOLOX/Tesseract

Key features:
- UnstructuredReader with hi_res strategy (ADR-004)
- Simple diskcache for document caching
- Basic spaCy model management with auto-download
- Comprehensive error handling with loguru
"""

import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any

import diskcache
from llama_index.core import Document
from llama_index.readers.file import UnstructuredReader
from loguru import logger

from src.config.app_settings import app_settings

# Simple document cache using diskcache
_document_cache = diskcache.Cache("./cache/documents")


def _get_file_hash(file_path: str) -> str:
    """Generate SHA-256 hash for file content."""
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()
    except (OSError, PermissionError) as e:
        logger.warning("Could not hash file %s: %s", file_path, e)
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
    _document_cache.set(file_hash, documents, expire=app_settings.cache_expiry_seconds)


def load_documents_unstructured(file_path: str | Path) -> list[Document]:
    """Load documents using UnstructuredReader with hi_res strategy (ADR-004).

    Uses Unstructured library for offline multimodal document processing including
    text, images, and tables with OCR support via YOLOX/Tesseract.

    Args:
        file_path: Path to document file to process

    Returns:
        List of Document objects with multimodal content and metadata

    Raises:
        RuntimeError: If document loading fails after basic retries

    Note:
        Follows ADR-004 specification for offline document parsing using
        UnstructuredReader.load_data(file_path, strategy='hi_res')
    """
    file_path = str(file_path)

    # Check cache first
    cached_docs = _get_cached(file_path)
    if cached_docs:
        logger.info("Using cached document for %s", Path(file_path).name)
        return cached_docs

    logger.info("Loading document with UnstructuredReader: %s", Path(file_path).name)

    try:
        # Validate file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Initialize UnstructuredReader (ADR-004 compliant)
        reader = UnstructuredReader()

        # Load with hi_res strategy for multimodal extraction (ADR-004)
        documents = reader.load_data(
            file=file_path,
            strategy=getattr(app_settings, "parse_strategy", "hi_res"),
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
                    "strategy": getattr(app_settings, "parse_strategy", "hi_res"),
                }
            )

        # Cache results
        _cache_documents(file_path, documents)

        logger.success(
            "Loaded %d elements from %s", len(documents), Path(file_path).name
        )
        return documents

    except (
        FileNotFoundError,
        PermissionError,
        ValueError,
        ImportError,
        RuntimeError,
    ) as e:
        error_msg = f"Failed to load document {Path(file_path).name}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def load_documents_from_directory(directory_path: str | Path) -> list[Document]:
    """Load all supported documents from a directory.

    Args:
        directory_path: Path to directory containing documents

    Returns:
        List of all loaded Document objects

    Raises:
        RuntimeError: If directory access fails
    """
    directory_path = Path(directory_path)

    if not directory_path.exists() or not directory_path.is_dir():
        raise RuntimeError(f"Directory not found: {directory_path}")

    # Supported file extensions
    supported_extensions = {".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md"}

    # Find all supported files
    files = []
    for ext in supported_extensions:
        files.extend(directory_path.glob(f"*{ext}"))
        files.extend(directory_path.glob(f"**/*{ext}"))  # Recursive search

    if not files:
        logger.warning("No supported documents found in %s", directory_path)
        return []

    logger.info("Found %d documents in %s", len(files), directory_path)

    # Load all documents
    all_documents = []
    for file_path in files:
        try:
            documents = load_documents_unstructured(file_path)
            all_documents.extend(documents)
        except (
            FileNotFoundError,
            PermissionError,
            ValueError,
            ImportError,
            RuntimeError,
        ) as e:
            logger.warning("Failed to load %s: %s", file_path.name, e)
            continue

    logger.success(
        "Loaded %d total elements from %d files", len(all_documents), len(files)
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
        "size_mb": round(stat.st_size / app_settings.bytes_to_mb_divisor, 2),
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
        logger.info("Cleared %d cached documents", count)
        return count
    except (OSError, PermissionError) as e:
        logger.error("Failed to clear cache: %s", e)
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
    except (OSError, PermissionError, AttributeError) as e:
        logger.error("Failed to get cache stats: %s", e)
        return {"error": str(e)}


# spaCy Model Management Functions


def ensure_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    """Ensure spaCy model is available, download if needed.

    Args:
        model_name: Name of the spaCy model to load (default: en_core_web_sm)

    Returns:
        Loaded spaCy Language model instance or None if fails
    """
    start_time = time.perf_counter()
    logger.info("Loading spaCy model: %s", model_name)

    try:
        import spacy

        # Try to load existing model first
        try:
            nlp = spacy.load(model_name)
            logger.success("spaCy model '%s' loaded successfully", model_name)

            duration = time.perf_counter() - start_time
            logger.info("spaCy model loaded in %.2fs", duration)
            return nlp

        except OSError:
            # Model not found locally, try to download
            logger.info(
                "spaCy model '%s' not found locally, downloading...", model_name
            )

            try:
                # Download with simple subprocess call
                import os

                download_cmd = ["python", "-m", "spacy", "download", model_name]

                # Try uv if available
                if any(
                    os.path.exists(os.path.join(path, "uv"))
                    for path in os.environ.get("PATH", "").split(os.pathsep)
                ):
                    download_cmd = [
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "spacy",
                        "download",
                        model_name,
                    ]

                subprocess.run(
                    download_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=app_settings.spacy_download_timeout,
                )

                logger.info("spaCy model '%s' downloaded successfully", model_name)

                # Try loading again after download
                nlp = spacy.load(model_name)
                logger.success("spaCy model '%s' loaded after download", model_name)

                duration = time.perf_counter() - start_time
                logger.info("spaCy model downloaded and loaded in %.2fs", duration)
                return nlp

            except (
                subprocess.TimeoutExpired,
                subprocess.CalledProcessError,
                OSError,
            ) as e:
                logger.error("Failed to download spaCy model '%s': %s", model_name, e)
                return None

    except ImportError:
        logger.error("spaCy is not installed. Install with: pip install spacy")
        return None
    except (RuntimeError, ValueError) as e:
        logger.error("Unexpected error loading spaCy model '%s': %s", model_name, e)
        return None


def extract_entities_with_spacy(
    text: str, model_name: str = "en_core_web_sm"
) -> list[dict[str, Any]]:
    """Extract named entities from text using spaCy.

    Args:
        text: Text to extract entities from
        model_name: spaCy model to use for extraction

    Returns:
        List of dictionaries with entity information
    """
    nlp = ensure_spacy_model(model_name)

    if nlp is None:
        logger.warning("spaCy model not available, returning empty entities")
        return []

    try:
        doc = nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": app_settings.default_entity_confidence,  # spaCy
                }
            )

        logger.info("Extracted %d entities from text", len(entities))
        return entities

    except (ValueError, RuntimeError, AttributeError) as e:
        logger.error("Failed to extract entities: %s", e)
        return []


def extract_relationships_with_spacy(
    text: str, model_name: str = "en_core_web_sm"
) -> list[dict[str, Any]]:
    """Extract relationships from text using spaCy dependency parsing.

    Args:
        text: Text to extract relationships from
        model_name: spaCy model to use for parsing

    Returns:
        List of dictionaries with relationship information
    """
    nlp = ensure_spacy_model(model_name)

    if nlp is None:
        logger.warning("spaCy model not available, returning empty relationships")
        return []

    try:
        doc = nlp(text)
        relationships = []

        for token in doc:
            # Focus on meaningful relationships (subject-verb-object patterns)
            if token.dep_ in ["nsubj", "dobj", "pobj"] and token.head.pos_ == "VERB":
                relationships.append(
                    {
                        "subject": token.text,
                        "relation": token.head.text,
                        "object": token.head.text,
                        "dependency": token.dep_,
                        "start": token.idx,
                        "end": token.idx + len(token.text),
                    }
                )

        logger.info("Extracted %d relationships from text", len(relationships))
        return relationships

    except (ValueError, RuntimeError, AttributeError) as e:
        logger.error("Failed to extract relationships: %s", e)
        return []


def create_knowledge_graph_data(
    text: str, model_name: str = "en_core_web_sm"
) -> dict[str, Any]:
    """Create knowledge graph data from text using spaCy.

    Args:
        text: Text to process
        model_name: spaCy model to use

    Returns:
        Dictionary with entities and relationships for knowledge graph
    """
    entities = extract_entities_with_spacy(text, model_name)
    relationships = extract_relationships_with_spacy(text, model_name)

    return {
        "entities": entities,
        "relationships": relationships,
        "text_length": len(text),
        "processed_at": time.time(),
    }


# Backwards compatibility aliases
load_documents_llama = load_documents_unstructured  # Legacy alias
