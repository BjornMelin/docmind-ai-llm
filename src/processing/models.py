"""Processing-specific Pydantic models for DocMind AI Document Processing Pipeline.

This module contains data models specifically designed for the document processing
pipeline, following ADR-009 requirements for direct Unstructured.io integration,
semantic chunking, and BGE-M3 embeddings.

HYBRID MODEL ORGANIZATION:
Following the hybrid model organization strategy, these models are colocated within
the processing domain because they are:
- Tightly coupled to document processing workflows
- Specific to Unstructured.io partition() operations
- Domain-specific to document processing pipeline
- Used primarily within the processing modules

Models:
    ProcessingStrategy: Enum for document processing strategies
    DocumentElement: Structured representation of unstructured.io elements
    ProcessingResult: Result of document processing operations
    ProcessingError: Custom exception for processing errors

Architecture Decision:
    These models are placed within the processing module following domain-driven
    design principles. They are domain-specific and tightly coupled to the
    document processing system, making this the most appropriate location per
    the hybrid organization strategy.
"""

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProcessingStrategy(str, Enum):
    """Processing strategies for different document types."""

    HI_RES = "hi_res"
    FAST = "fast"
    OCR_ONLY = "ocr_only"


class DocumentElement(BaseModel):
    """Structured representation of a document element from unstructured.io."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = Field(description="Element text content")
    category: str = Field(
        description="Element category (Title, NarrativeText, Table, Image, etc.)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Complete element metadata"
    )

    def __init__(self, **data):
        """Initialize DocumentElement from unstructured.io element."""
        super().__init__(**data)


class ProcessingResult(BaseModel):
    """Result of document processing operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    elements: list[DocumentElement] = Field(description="Processed document elements")
    processing_time: float = Field(description="Processing time in seconds")
    strategy_used: ProcessingStrategy = Field(description="Processing strategy applied")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Processing metadata"
    )
    document_hash: str = Field(description="Document content hash for caching")

    @classmethod
    def create_hash_for_document(cls, file_path: str | Path) -> str:
        """Calculate unique hash for document caching.

        Args:
            file_path: Path to the document file

        Returns:
            SHA-256 hash string of file content and metadata
        """
        file_path = Path(file_path)

        # Hash file content + metadata for cache key
        hasher = hashlib.sha256()

        # Include file content
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        # Include file metadata
        stat = file_path.stat()
        metadata = f"{file_path.name}:{stat.st_size}:{stat.st_mtime}".encode()
        hasher.update(metadata)

        return hasher.hexdigest()


class ProcessingError(Exception):
    """Custom exception for document processing errors."""

    pass
