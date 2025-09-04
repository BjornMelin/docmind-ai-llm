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

from pathlib import Path
from typing import Any

from loguru import logger

from src.core.spacy_manager import get_spacy_manager
from src.processing.document_processor import DocumentProcessor, ProcessingError


async def load_documents_unstructured(
    file_paths: list[str | Path], settings: Any | None = None
) -> list[Any]:
    """Load documents using ADR-009 compliant ResilientDocumentProcessor.

    Args:
        file_paths: List of file paths to process
        settings: Optional DocMind settings

    Returns:
        List of processed documents
    """
    logger.info(
        f"Processing {len(file_paths)} documents via ResilientDocumentProcessor"
    )

    processor = DocumentProcessor(settings)
    results = []

    for file_path in file_paths:
        try:
            result = await processor.process_document_async(file_path)
            results.append(result)
        except (ProcessingError, ValueError, OSError) as e:
            logger.error(f"Failed to process {file_path}: {e!s}")
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
    processor = DocumentProcessor()
    try:
        strategy = processor.get_strategy_for_file(file_path)
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
    logger.info("Clearing document processing cache (duckdb file)")
    try:
        from src.config.settings import settings as app_settings

        cache_dir = Path(getattr(app_settings, "cache_dir", "./cache"))
        cache_db = cache_dir / "docmind.duckdb"
        if cache_db.exists():
            cache_db.unlink()
        logger.info("Document cache cleared (duckdb file removed if present)")
        return True
    except (OSError, RuntimeError) as e:
        logger.warning(f"Failed to clear document cache: {e}")
        return False


async def get_cache_stats() -> dict[str, Any]:
    """Get document processing cache statistics.

    Returns:
        Cache statistics dictionary
    """
    try:
        from src.config.settings import settings as app_settings

        cache_dir = Path(getattr(app_settings, "cache_dir", "./cache"))
        cache_db = cache_dir / "docmind.duckdb"
        stats = {
            "cache_type": "duckdb_kvstore",
            "db_path": str(cache_db),
            "total_documents": -1,
        }
        logger.debug(f"Cache statistics: type={stats.get('cache_type', 'unknown')}")
        logger.debug(f"Cache path: {stats.get('db_path')}")
        return stats
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning(f"Failed to read cache stats: {e}")
        return {"cache_type": "duckdb_kvstore", "error": str(e)}


def ensure_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    """Ensure spaCy model is available and loaded using SpacyManager architecture.

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


def clear_document_cache_sync() -> bool:
    """Synchronous wrapper for cache clearing.

    Returns:
        True if cache was cleared successfully
    """
    import asyncio

    return asyncio.run(clear_document_cache())


def get_cache_stats_sync() -> dict[str, Any]:
    """Synchronous wrapper for cache stats.

    Returns:
        Cache statistics dictionary
    """
    import asyncio

    return asyncio.run(get_cache_stats())


def extract_entities_with_spacy(text: str, nlp: Any = None) -> list[dict[str, Any]]:
    """Extract entities using spaCy.

    Args:
        text: Text to process
        nlp: Optional spaCy nlp model, defaults to ensure_spacy_model()

    Returns:
        List of entities with text, label, start, end positions
    """
    if nlp is None:
        nlp = ensure_spacy_model()

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
                    "confidence": getattr(ent, "confidence", 0.8),  # Default confidence
                }
            )
        return entities
    except (ValueError, OSError, AttributeError) as e:
        logger.warning(f"Failed to extract entities with spaCy: {e}")
        return []


def extract_relationships_with_spacy(
    text: str, nlp: Any = None
) -> list[dict[str, Any]]:
    """Extract relationships using spaCy.

    Args:
        text: Text to process
        nlp: Optional spaCy nlp model, defaults to ensure_spacy_model()

    Returns:
        List of relationships with source, target, type information
    """
    if nlp is None:
        nlp = ensure_spacy_model()

    try:
        doc = nlp(text)
        relationships = []

        # Simple relationship extraction based on dependency parsing
        for token in doc:
            if token.dep_ in ["nsubj", "dobj", "pobj"] and token.head.pos_ == "VERB":
                relationships.append(
                    {
                        "source": token.text,
                        "target": token.head.text,
                        "type": token.dep_.upper(),
                        "confidence": 0.7,
                    }
                )

        return relationships
    except (ValueError, OSError, AttributeError) as e:
        logger.warning(f"Failed to extract relationships with spaCy: {e}")
        return []


def create_knowledge_graph_data(
    text_or_entities: str | list, nlp_or_relationships: Any = None
) -> dict[str, Any]:
    """Create knowledge graph structure.

    Args:
        text_or_entities: Either text to process or list of entities
        nlp_or_relationships: Either spaCy nlp model or list of relationships

    Returns:
        Dictionary with entities, relationships, and metadata
    """
    try:
        # Handle two different calling patterns from tests
        if isinstance(text_or_entities, str):
            # Called with text and nlp model
            text = text_or_entities
            nlp = nlp_or_relationships or ensure_spacy_model()
            entities = extract_entities_with_spacy(text, nlp)
            relationships = extract_relationships_with_spacy(text, nlp)
        else:
            # Called with entities and relationships lists
            entities = text_or_entities or []
            relationships = nlp_or_relationships or []
            text = ""  # No text in this mode

        return {
            "entities": entities,
            "relationships": relationships,
            "metadata": {
                "entity_count": len(entities),
                "relationship_count": len(relationships),
                "text_length": len(text) if isinstance(text_or_entities, str) else 0,
                "processing_method": "spacy_knowledge_graph",
            },
        }
    except (ValueError, OSError, AttributeError) as e:
        logger.warning(f"Failed to create knowledge graph data: {e}")
        return {
            "entities": [],
            "relationships": [],
            "metadata": {
                "entity_count": 0,
                "relationship_count": 0,
                "text_length": 0,
                "processing_method": "spacy_knowledge_graph",
                "error": str(e),
            },
        }


# Convenient function aliases
load_documents = load_documents_unstructured  # Common alias
get_doc_info = get_document_info  # Short alias
clear_cache = clear_document_cache  # Short alias
