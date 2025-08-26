"""DocMind AI Semantic Chunking Module.

This module implements semantic chunking with direct chunk_by_title() integration
from Unstructured.io for intelligent boundary detection and multipage section
handling following ADR-009 requirements.

Components:
    unstructured_chunker: SemanticChunker with chunk_by_title implementation
    models: Chunking-specific Pydantic models

Key Features:
- Direct unstructured.chunking.title.chunk_by_title() integration
- Intelligent semantic boundary detection
- Configurable chunking parameters (max_characters=1500)
- Multipage section handling with hierarchy preservation
- Boundary accuracy measurement and optimization
"""

from src.processing.chunking.models import (
    BoundaryDetection,
    ChunkingError,
    ChunkingParameters,
    ChunkingResult,
    SemanticChunk,
)
from src.processing.chunking.unstructured_chunker import SemanticChunker

__all__ = [
    "SemanticChunker",
    "BoundaryDetection",
    "ChunkingParameters",
    "SemanticChunk",
    "ChunkingResult",
    "ChunkingError",
]
