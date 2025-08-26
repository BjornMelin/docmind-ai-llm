"""Document processing utilities for DocMind AI.

This module provides convenient wrapper functions for document processing
operations using the ADR-009 compliant architecture.

Available Functions:
- load_documents_unstructured: Redirects to ResilientDocumentProcessor
- load_documents_from_directory: Batch document processing
- get_document_info: Document metadata extraction
- clear_document_cache: Cache clearing operations
- get_cache_stats: Cache statistics
- ensure_spacy_model: spaCy model management
"""

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from src.cache import create_cache_manager
from src.core.infrastructure.spacy_manager import get_spacy_manager
from src.processing.document_processor import create_resilient_processor


async def load_documents_unstructured(
    file_paths: list[str | Path], settings: Any | None = None
) -> list[Any]:
    """Load documents using ADR-009 compliant ResilientDocumentProcessor.

    This function provides compatibility with the legacy interface while
    using the new processing architecture.

    Args:
        file_paths: List of file paths to process
        settings: Optional DocMind settings

    Returns:
        List of processed documents
    """
    logger.info(
        f"Processing {len(file_paths)} documents via ResilientDocumentProcessor"
    )

    processor = create_resilient_processor(settings)
    results = []

    for file_path in file_paths:
        try:
            result = await processor.process_document_async(file_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            # Continue processing other files
            continue

    logger.info(f"Successfully processed {len(results)}/{len(file_paths)} documents")
    return results


async def load_documents_from_directory(
    directory_path: str | Path,
    settings: Any | None = None,
    recursive: bool = True,
    supported_extensions: set[str] | None = None,
) -> list[Any]:
    """Load all supported documents from a directory.

    Args:
        directory_path: Path to directory containing documents
        settings: Optional DocMind settings
        recursive: Whether to search subdirectories
        supported_extensions: Set of file extensions to process

    Returns:
        List of processed documents
    """
    directory_path = Path(directory_path)

    if supported_extensions is None:
        # Default extensions from ResilientDocumentProcessor
        supported_extensions = {
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".html",
            ".htm",
            ".txt",
            ".md",
            ".rtf",
            ".jpg",
            ".jpeg",
            ".png",
            ".tiff",
            ".bmp",
        }

    # Find all supported files
    file_paths = []
    if recursive:
        for ext in supported_extensions:
            file_paths.extend(directory_path.rglob(f"*{ext}"))
    else:
        for ext in supported_extensions:
            file_paths.extend(directory_path.glob(f"*{ext}"))

    logger.info(f"Found {len(file_paths)} supported files in {directory_path}")

    # Process all files
    return await load_documents_unstructured(file_paths, settings)


def get_document_info(file_path: str | Path) -> dict[str, Any]:
    """Get document information and metadata.

    Args:
        file_path: Path to the document file

    Returns:
        Dictionary with document information
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    stat = file_path.stat()

    # Basic file information
    info = {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "file_extension": file_path.suffix.lower(),
        "file_size_bytes": stat.st_size,
        "file_size_mb": stat.st_size / (1024 * 1024),
        "created_time": stat.st_ctime,
        "modified_time": stat.st_mtime,
        "is_readable": file_path.is_file() and stat.st_size > 0,
    }

    # Determine processing strategy
    processor = create_resilient_processor()
    try:
        strategy = processor._get_strategy_for_file(file_path)
        info["processing_strategy"] = strategy.value
        info["supported"] = True
    except ValueError:
        info["processing_strategy"] = None
        info["supported"] = False

    return info


async def clear_document_cache() -> bool:
    """Clear document processing cache.

    Returns:
        True if cache was cleared successfully
    """
    logger.info("Clearing document processing cache")

    cache_manager = create_cache_manager()

    # Simple cache has only one layer
    success = await cache_manager.clear_cache()

    if success:
        logger.info("Document cache cleared successfully")
    else:
        logger.warning("Failed to clear document cache")

    return success


async def get_cache_stats() -> dict[str, Any]:
    """Get document processing cache statistics.

    Returns:
        Cache statistics dictionary
    """
    cache_manager = create_cache_manager()
    stats = await cache_manager.get_cache_stats()

    logger.debug(f"Cache statistics: type={stats.get('cache_type', 'unknown')}")

    return stats


def ensure_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    """Ensure spaCy model is available and loaded.

    This function provides compatibility with the legacy interface while
    using the new SpacyManager architecture.

    Args:
        model_name: Name of the spaCy model to load

    Returns:
        Loaded spaCy language model
    """
    logger.info(f"Ensuring spaCy model: {model_name}")

    spacy_manager = get_spacy_manager()
    nlp = spacy_manager.ensure_model(model_name)

    logger.info(f"spaCy model '{model_name}' loaded successfully")
    return nlp


# Convenient function aliases
load_documents = load_documents_unstructured  # Common alias
get_doc_info = get_document_info  # Short alias
clear_cache = clear_document_cache  # Short alias


# Async wrapper for sync functions that need async context
def _run_async_in_sync_context(coro):
    """Run async coroutine in sync context."""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to use asyncio.create_task
            # But for compatibility, we'll just warn and return empty result
            logger.warning(
                "Cannot run async function in sync context with running loop"
            )
            return None
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(coro)


# Synchronous wrappers for convenience
def clear_document_cache_sync() -> int:
    """Synchronous wrapper for clear_document_cache."""
    result = _run_async_in_sync_context(clear_document_cache())
    return result if result is not None else 0


def get_cache_stats_sync() -> dict[str, Any]:
    """Synchronous wrapper for get_cache_stats."""
    result = _run_async_in_sync_context(get_cache_stats())
    return result if result is not None else {}


# Knowledge Graph Functions
def extract_entities_with_spacy(
    text: str, nlp_model: Any = None
) -> list[dict[str, Any]]:
    """Extract entities from text using spaCy.

    This is a compatibility function for legacy knowledge graph functionality.
    In ADR-009 compliant architecture, this functionality should be implemented
    using the new processing pipeline.

    Args:
        text: Input text for entity extraction
        nlp_model: Optional spaCy model (loads default if None)

    Returns:
        List of entity dictionaries with text, label, start, end fields
    """
    if nlp_model is None:
        nlp_model = ensure_spacy_model()

    try:
        doc = nlp_model(text)
        entities = []

        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": getattr(ent, "score", 1.0),  # Confidence if available
                }
            )

        logger.debug(f"Extracted {len(entities)} entities from text")
        return entities

    except Exception as e:
        logger.error(f"Failed to extract entities: {str(e)}")
        return []


def extract_relationships_with_spacy(
    text: str, nlp_model: Any = None
) -> list[dict[str, str]]:
    """Extract relationships from text using spaCy dependency parsing.

    This is a compatibility function for legacy knowledge graph functionality.
    Uses basic dependency parsing to identify subject-verb-object relationships.

    Args:
        text: Input text for relationship extraction
        nlp_model: Optional spaCy model (loads default if None)

    Returns:
        List of relationship dictionaries with subject, predicate, object fields
    """
    if nlp_model is None:
        nlp_model = ensure_spacy_model()

    try:
        doc = nlp_model(text)
        relationships = []

        # Simple subject-verb-object extraction using dependency parsing
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subject = None
                obj = None

                # Find subject
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = child.text
                    elif child.dep_ in ("dobj", "pobj"):
                        obj = child.text

                if subject and obj:
                    relationships.append(
                        {"subject": subject, "predicate": token.text, "object": obj}
                    )

        logger.debug(f"Extracted {len(relationships)} relationships from text")
        return relationships

    except Exception as e:
        logger.error(f"Failed to extract relationships: {str(e)}")
        return []


def create_knowledge_graph_data(text: str, nlp_model: Any = None) -> dict[str, Any]:
    """Create knowledge graph data from text using spaCy.

    This is a compatibility function for legacy knowledge graph functionality.
    Combines entity extraction and relationship extraction.

    Args:
        text: Input text for knowledge graph creation
        nlp_model: Optional spaCy model (loads default if None)

    Returns:
        Dictionary with 'entities' and 'relationships' keys
    """
    logger.info("Creating knowledge graph data from text")

    try:
        entities = extract_entities_with_spacy(text, nlp_model)
        relationships = extract_relationships_with_spacy(text, nlp_model)

        result = {
            "entities": entities,
            "relationships": relationships,
            "metadata": {
                "text_length": len(text),
                "entity_count": len(entities),
                "relationship_count": len(relationships),
                "processing_method": "spacy_knowledge_graph",
            },
        }

        logger.info(
            f"Generated knowledge graph: {len(entities)} entities, "
            f"{len(relationships)} relationships"
        )
        return result

    except Exception as e:
        logger.error(f"Failed to create knowledge graph data: {str(e)}")
        return {
            "entities": [],
            "relationships": [],
            "metadata": {
                "error": str(e),
                "processing_method": "spacy_knowledge_graph",
            },
        }
