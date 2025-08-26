"""Chunking-specific Pydantic models for DocMind AI Semantic Chunking Pipeline.

This module contains data models specifically designed for the semantic chunking
pipeline using direct chunk_by_title() integration from Unstructured.io, following
ADR-009 requirements for intelligent boundary detection and multipage section handling.

HYBRID MODEL ORGANIZATION:
Following the hybrid model organization strategy, these models are colocated within
the chunking subdomain because they are:
- Tightly coupled to chunk_by_title() operations
- Specific to Unstructured.io semantic chunking
- Domain-specific to chunking workflows
- Used primarily within the chunking modules

Models:
    BoundaryDetection: Enum for semantic boundary detection strategies
    ChunkingParameters: Configuration parameters for semantic chunking
    SemanticChunk: Semantic chunk with boundary detection and metadata
    ChunkingResult: Result of semantic chunking operations
    ChunkingError: Custom exception for chunking errors

Architecture Decision:
    These models are placed within the chunking submodule following domain-driven
    design principles. They are specific to the chunking domain and tightly coupled
    to the semantic chunking system, making this the most appropriate location per
    the hybrid organization strategy.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BoundaryDetection(str, Enum):
    """Semantic boundary detection strategies."""

    TITLE_BASED = "title"
    CONTENT_BASED = "content"
    HYBRID = "hybrid"


class ChunkingParameters(BaseModel):
    """Configuration parameters for semantic chunking."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_characters: int = Field(
        default=1500, ge=100, le=10000, description="Maximum characters per chunk"
    )
    new_after_n_chars: int = Field(
        default=1200, ge=100, le=8000, description="Minimum characters before new chunk"
    )
    combine_text_under_n_chars: int = Field(
        default=500, ge=50, le=2000, description="Combine text under this threshold"
    )
    multipage_sections: bool = Field(
        default=True, description="Allow sections to span multiple pages"
    )
    boundary_detection: BoundaryDetection = Field(
        default=BoundaryDetection.TITLE_BASED, description="Boundary detection strategy"
    )
    preserve_section_hierarchy: bool = Field(
        default=True, description="Preserve document section hierarchy"
    )
    include_orig_elements: bool = Field(
        default=True, description="Include original elements in metadata"
    )


class SemanticChunk(BaseModel):
    """Semantic chunk with boundary detection and metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = Field(description="Chunk text content")
    category: str = Field(default="CompositeElement", description="Chunk category")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Complete chunk metadata"
    )
    section_title: str | None = Field(
        default=None, description="Associated section title"
    )
    chunk_index: int = Field(description="Index of chunk in document")
    semantic_boundaries: dict[str, Any] = Field(
        default_factory=dict, description="Boundary detection metadata"
    )
    original_elements_count: int = Field(
        default=0, description="Number of original elements combined"
    )


class ChunkingResult(BaseModel):
    """Result of semantic chunking operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunks: list[SemanticChunk] = Field(description="Semantic chunks")
    total_elements: int = Field(
        description="Total number of original elements processed"
    )
    boundary_accuracy: float = Field(
        description="Boundary detection accuracy score (0.0-1.0)"
    )
    processing_time: float = Field(description="Chunking processing time in seconds")
    parameters: ChunkingParameters = Field(description="Parameters used for chunking")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Chunking result metadata"
    )


class ChunkingError(Exception):
    """Custom exception for semantic chunking errors."""

    pass
