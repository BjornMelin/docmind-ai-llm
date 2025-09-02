"""Semantic Chunker with direct Unstructured.io chunk_by_title integration.

This module implements ADR-009 compliant semantic chunking using direct
chunk_by_title() calls from unstructured.io for intelligent boundary detection
and multipage section handling.

Key Features:
- Direct unstructured.chunking.title.chunk_by_title() integration
- Intelligent semantic boundary detection (title-based, content-based, hybrid)
- Configurable chunking parameters (max_characters, combine thresholds)
- Multipage section handling with hierarchy preservation
- Boundary accuracy measurement and optimization
- Async processing for non-blocking operations
"""

import asyncio
import time
from enum import Enum
from typing import Any

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from unstructured.chunking.title import chunk_by_title

from src.processing.chunking.models import (
    ChunkingParameters,
    ChunkingResult,
    SemanticChunk,
)

try:
    from dataclasses import dataclass
except ImportError:
    # Fallback for older Python versions
    def dataclass(cls):
        """Fallback dataclass decorator for compatibility."""
        return cls


@dataclass
class ElementMetadata:
    """Metadata adapter for unstructured element compatibility.

    This adapter provides metadata attributes expected by unstructured.io's
    chunk_by_title function. It converts DocumentElement metadata to the
    format required by the unstructured library.
    """

    page_number: int = 1
    element_id: str | None = None
    parent_id: str | None = None
    filename: str | None = None
    coordinates: Any | None = None
    section_title: str | None = None
    text_as_html: str | None = None
    image_path: str | None = None


@dataclass
class ElementAdapter:
    """Element adapter for unstructured.io compatibility.

    This adapter wraps DocumentElement objects to provide the interface
    expected by unstructured.io's chunk_by_title function. It converts
    internal document representations to unstructured-compatible format.
    """

    text: str = ""
    category: str = "NarrativeText"
    metadata: ElementMetadata | None = None

    def __post_init__(self) -> None:
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = ElementMetadata()


class BoundaryDetection(str, Enum):
    """Semantic boundary detection strategies."""

    TITLE_BASED = "title"
    CONTENT_BASED = "content"
    HYBRID = "hybrid"


class ChunkingError(Exception):
    """Custom exception for semantic chunking errors."""


class SemanticChunker:
    """Semantic chunker with direct unstructured.io chunk_by_title integration.

    This chunker implements ADR-009 requirements for semantic boundary detection
    using direct chunk_by_title() calls from unstructured.io with no wrapper
    abstractions. It provides:

    - Direct chunk_by_title() calls with intelligent boundary detection
    - Multipage section handling with hierarchy preservation
    - Configurable chunking parameters for different use cases
    - Boundary accuracy measurement and optimization
    - Async processing for non-blocking operations
    """

    def __init__(self, settings: Any | None = None):
        """Initialize SemanticChunker.

        Args:
            settings: DocMind configuration settings. Uses default settings if None.
        """
        from src.config import settings as default_settings

        self.settings = settings or default_settings

        # Coerce defaults to a valid relation: combine_under < new_after < max
        raw_max = int(getattr(self.settings.processing, "chunk_size", 1500))
        raw_new_after = int(
            getattr(
                self.settings.processing, "new_after_n_chars", max(100, raw_max - 300)
            )
        )
        raw_combine_under = int(
            getattr(
                self.settings.processing,
                "combine_text_under_n_chars",
                max(50, raw_new_after - 300),
            )
        )

        max_characters = max(100, raw_max)
        new_after_n_chars = max(100, min(raw_new_after, max_characters - 1))
        combine_text_under_n_chars = max(
            0, min(raw_combine_under, new_after_n_chars - 1)
        )

        # Default chunking parameters from (coerced) settings
        self.default_parameters = ChunkingParameters(
            max_characters=max_characters,
            new_after_n_chars=new_after_n_chars,
            combine_text_under_n_chars=combine_text_under_n_chars,
            multipage_sections=bool(
                getattr(self.settings.processing, "multipage_sections", True)
            ),
            boundary_detection=BoundaryDetection.TITLE_BASED,
        )

        logger.info(
            "SemanticChunker init: max_chars={}, new_after={}, combine_under={}",
            self.default_parameters.max_characters,
            self.default_parameters.new_after_n_chars,
            self.default_parameters.combine_text_under_n_chars,
        )

    def _convert_document_elements_to_unstructured(
        self, elements: list[Any]
    ) -> list[Any]:
        """Convert DocumentElement objects to format expected by chunk_by_title.

        Args:
            elements: List of DocumentElement objects

        Returns:
            List of objects compatible with unstructured chunk_by_title
        """
        converted_elements = []

        for element in elements:
            # Extract text and category
            text = element.text if hasattr(element, "text") else str(element)
            category = (
                element.category if hasattr(element, "category") else "NarrativeText"
            )

            # Create metadata adapter
            metadata = ElementMetadata()

            # Set metadata if available
            if hasattr(element, "metadata") and element.metadata:
                for key, value in element.metadata.items():
                    if hasattr(metadata, key):
                        setattr(metadata, key, value)
            else:
                # Create minimal metadata
                metadata.page_number = 1
                metadata.element_id = f"elem_{len(converted_elements)}"
                metadata.filename = "document"

            # Create element adapter
            element_adapter = ElementAdapter(
                text=text, category=category, metadata=metadata
            )

            converted_elements.append(element_adapter)

        return converted_elements

    def _calculate_boundary_accuracy(
        self,
        chunks: list[Any],
        original_elements: list[Any],
        parameters: ChunkingParameters,
    ) -> float:
        """Calculate semantic boundary detection accuracy.

        Args:
            chunks: List of semantic chunks
            original_elements: Original document elements
            parameters: Chunking parameters used

        Returns:
            Boundary accuracy score (0.0-1.0)
        """
        if not chunks or not original_elements:
            return 0.0

        # Accuracy metrics based on boundary detection strategy
        if parameters.boundary_detection == BoundaryDetection.TITLE_BASED:
            # Count title-based boundaries that were preserved
            title_elements = [
                e
                for e in original_elements
                if hasattr(e, "category") and "title" in str(e.category).lower()
            ]
            title_boundaries_preserved = 0

            for chunk in chunks:
                chunk_metadata = getattr(chunk, "metadata", None)
                if (
                    chunk_metadata
                    and hasattr(chunk_metadata, "section_title")
                    and chunk_metadata.section_title
                ):
                    title_boundaries_preserved += 1

            if title_elements:
                accuracy = min(1.0, title_boundaries_preserved / len(title_elements))
            else:
                # If no titles, base accuracy on chunk size consistency
                avg_chunk_size = sum(
                    len(str(getattr(c, "text", ""))) for c in chunks
                ) / len(chunks)
                target_size = parameters.max_characters
                accuracy = max(
                    0.0, 1.0 - abs(avg_chunk_size - target_size) / target_size
                )
        else:
            # For other strategies, use chunk size distribution as proxy
            chunk_sizes = [len(str(getattr(c, "text", ""))) for c in chunks]
            if chunk_sizes:
                avg_size = sum(chunk_sizes) / len(chunk_sizes)
                size_variance = sum(
                    (size - avg_size) ** 2 for size in chunk_sizes
                ) / len(chunk_sizes)
                # Lower variance indicates better boundary detection
                accuracy = max(
                    0.0, 1.0 - (size_variance / (parameters.max_characters**2))
                )
            else:
                accuracy = 0.0

        # Ensure accuracy is between 0.0 and 1.0
        return max(0.0, min(1.0, accuracy))

    def _convert_unstructured_chunks_to_semantic(
        self, chunks: list[Any], parameters: ChunkingParameters
    ) -> list[SemanticChunk]:
        """Convert unstructured chunks to SemanticChunk objects.

        Args:
            chunks: Raw chunks from chunk_by_title
            parameters: Chunking parameters used

        Returns:
            List of SemanticChunk objects with complete metadata
        """
        semantic_chunks = []

        for i, chunk in enumerate(chunks):
            # Extract chunk metadata
            chunk_metadata = {}
            if hasattr(chunk, "metadata") and chunk.metadata:
                chunk_metadata = {
                    "page_number": getattr(chunk.metadata, "page_number", None),
                    "element_id": getattr(chunk.metadata, "element_id", None),
                    "parent_id": getattr(chunk.metadata, "parent_id", None),
                    "filename": getattr(chunk.metadata, "filename", None),
                    "coordinates": getattr(chunk.metadata, "coordinates", None),
                    "section_title": getattr(chunk.metadata, "section_title", None),
                    "chunk_index": i,
                }

                # Remove None values
                chunk_metadata = {
                    k: v for k, v in chunk_metadata.items() if v is not None
                }

            # Determine section title
            section_title = None
            if chunk_metadata.get("section_title"):
                section_title = chunk_metadata["section_title"]
            elif hasattr(chunk, "metadata") and hasattr(
                chunk.metadata, "section_title"
            ):
                section_title = chunk.metadata.section_title

            # Calculate semantic boundary metadata
            semantic_boundaries = {
                "boundary_type": parameters.boundary_detection.value,
                "chunk_size": len(str(chunk.text)),
                "meets_size_target": parameters.new_after_n_chars
                <= len(str(chunk.text))
                <= parameters.max_characters,
                "section_based": section_title is not None,
            }

            semantic_chunk = SemanticChunk(
                text=str(chunk.text) if chunk.text else "",
                category=str(chunk.category)
                if hasattr(chunk, "category")
                else "CompositeElement",
                metadata=chunk_metadata,
                section_title=section_title,
                chunk_index=i,
                semantic_boundaries=semantic_boundaries,
                # chunk_by_title typically combines multiple elements
                original_elements_count=1,
            )

            semantic_chunks.append(semantic_chunk)

        return semantic_chunks

    @retry(
        retry=retry_if_exception_type((ChunkingError, ValueError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
    )
    async def chunk_elements_async(
        self, elements: list[Any], parameters: ChunkingParameters | None = None
    ) -> ChunkingResult:
        """Chunk document elements asynchronously using semantic boundaries.

        Args:
            elements: List of DocumentElement objects to chunk
            parameters: Optional chunking parameters. Uses defaults if None.

        Returns:
            ChunkingResult with semantic chunks and metadata

        Raises:
            ChunkingError: If chunking operation fails
            ValueError: If parameters are invalid
        """
        start_time = time.time()
        chunk_parameters = parameters or self.default_parameters

        logger.info(
            "Chunking {} elements with strategy: {}",
            len(elements),
            chunk_parameters.boundary_detection,
        )

        if not elements:
            logger.warning("No elements provided for chunking")
            return ChunkingResult(
                chunks=[],
                total_elements=0,
                boundary_accuracy=0.0,
                processing_time=time.time() - start_time,
                parameters=chunk_parameters,
                metadata={"warning": "No elements provided"},
            )

        try:
            # Convert elements to format expected by chunk_by_title
            unstructured_elements = self._convert_document_elements_to_unstructured(
                elements
            )

            # Validate parameter relationships
            if not (
                chunk_parameters.combine_text_under_n_chars
                < chunk_parameters.new_after_n_chars
                < chunk_parameters.max_characters
            ):
                raise ValueError(
                    "Invalid chunking params: combine_under < new_after < max_chars"
                )

            # Build chunk_by_title configuration
            chunk_config = {
                "max_characters": chunk_parameters.max_characters,
                "new_after_n_chars": chunk_parameters.new_after_n_chars,
                "combine_text_under_n_chars": (
                    chunk_parameters.combine_text_under_n_chars
                ),
                "multipage_sections": chunk_parameters.multipage_sections,
            }

            # Process chunks in thread pool to avoid blocking
            chunks = await asyncio.to_thread(
                self._chunk_by_title_sync, unstructured_elements, chunk_config
            )

            # Convert to SemanticChunk objects
            semantic_chunks = self._convert_unstructured_chunks_to_semantic(
                chunks, chunk_parameters
            )

            # Calculate boundary accuracy
            boundary_accuracy = self._calculate_boundary_accuracy(
                chunks, elements, chunk_parameters
            )

            processing_time = time.time() - start_time

            result = ChunkingResult(
                chunks=semantic_chunks,
                total_elements=len(elements),
                boundary_accuracy=boundary_accuracy,
                processing_time=processing_time,
                parameters=chunk_parameters,
                metadata={
                    "chunk_count": len(semantic_chunks),
                    "avg_chunk_size": sum(len(c.text) for c in semantic_chunks)
                    / len(semantic_chunks)
                    if semantic_chunks
                    else 0,
                    "boundary_strategy": chunk_parameters.boundary_detection.value,
                    "multipage_enabled": chunk_parameters.multipage_sections,
                },
            )

            logger.info(
                "Chunked {} elements into {} chunks in {:.2f}s (accuracy: {:.2f})",
                len(elements),
                len(semantic_chunks),
                processing_time,
                boundary_accuracy,
            )

            return result

        except ValueError as e:
            # Propagate parameter validation errors directly (tests expect ValueError)
            processing_time = time.time() - start_time
            logger.error(
                "Parameter validation failed after {:.2f}s: {}",
                processing_time,
                e,
            )
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Failed to chunk {} elements after {:.2f}s: {}",
                len(elements),
                processing_time,
                e,
            )
            raise ChunkingError(f"Semantic chunking failed: {e}") from e

    def _chunk_by_title_sync(
        self, elements: list[Any], config: dict[str, Any]
    ) -> list[Any]:
        """Synchronous chunking with direct chunk_by_title call.

        Args:
            elements: List of unstructured elements
            config: Chunking configuration

        Returns:
            List of chunked elements from chunk_by_title

        Raises:
            ChunkingError: If chunking fails
        """
        try:
            logger.debug("Calling chunk_by_title() with config: {}", config)

            # Direct unstructured.chunking.title.chunk_by_title() call
            chunks = chunk_by_title(elements=elements, **config)

            logger.debug("chunk_by_title returned {} chunks", len(chunks))
            return chunks

        except Exception as e:
            logger.error("chunk_by_title failed: {}", e)
            raise ChunkingError(f"Unstructured.io chunk_by_title failed: {e}") from e

    def optimize_parameters(
        self,
        elements: list[Any],
        target_chunk_count: int | None = None,
        target_accuracy: float = 0.85,
    ) -> ChunkingParameters:
        """Optimize chunking parameters for given elements.

        Args:
            elements: Sample elements to optimize for
            target_chunk_count: Target number of chunks (optional)
            target_accuracy: Target boundary accuracy (0.0-1.0)

        Returns:
            Optimized ChunkingParameters
        """
        if not elements:
            return self.default_parameters

        # Calculate total text length
        total_text_length = sum(len(str(getattr(e, "text", ""))) for e in elements)

        if target_chunk_count:
            # Optimize based on target chunk count
            target_chunk_size = total_text_length // target_chunk_count
            max_chars = min(3000, max(500, int(target_chunk_size * 1.2)))
            new_after_chars = min(
                max_chars - 300, max(300, int(target_chunk_size * 0.8))
            )
            combine_under_chars = min(
                new_after_chars - 100, max(100, int(target_chunk_size * 0.3))
            )
        else:
            # Use default optimization strategy
            max_chars = self.default_parameters.max_characters
            new_after_chars = self.default_parameters.new_after_n_chars
            combine_under_chars = self.default_parameters.combine_text_under_n_chars

        optimized_parameters = ChunkingParameters(
            max_characters=max_chars,
            new_after_n_chars=new_after_chars,
            combine_text_under_n_chars=combine_under_chars,
            multipage_sections=self.default_parameters.multipage_sections,
            boundary_detection=self.default_parameters.boundary_detection,
        )

        logger.info(
            "Optimized chunking params: max_chars={}, new_after={}, combine_under={}",
            max_chars,
            new_after_chars,
            combine_under_chars,
        )

        return optimized_parameters
