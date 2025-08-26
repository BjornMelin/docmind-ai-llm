"""Embedding-specific Pydantic models for DocMind AI BGE-M3 Embedding Pipeline.

This module contains data models specifically designed for the BGE-M3 embedding
operations, following ADR-009 requirements for 8K context unified embeddings
with dense, sparse, and ColBERT multi-vector representations.

HYBRID MODEL ORGANIZATION:
Following the hybrid model organization strategy, these models are colocated within
the embeddings subdomain because they are:
- Tightly coupled to BGE-M3 embedding operations
- Specific to FlagEmbedding BGE-M3 model workflows
- Domain-specific to embedding generation and processing
- Used primarily within the embeddings modules

Models:
    EmbeddingParameters: Configuration parameters for BGE-M3 embedding operations
    EmbeddingResult: Result of BGE-M3 embedding operations with dense/sparse/colbert
    EmbeddingError: Custom exception for embedding processing errors

Architecture Decision:
    These models are placed within the embeddings submodule following domain-driven
    design principles. They are specific to the embeddings domain and tightly coupled
    to the BGE-M3 embedding system, making this the most appropriate location per
    the hybrid organization strategy.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class EmbeddingParameters(BaseModel):
    """Configuration parameters for BGE-M3 embedding operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_length: int = Field(
        default=8192, ge=512, le=16384, description="Maximum token length (8K context)"
    )
    batch_size_gpu: int = Field(
        default=12,
        ge=1,
        le=32,
        description="GPU batch size for RTX 4090 "
        "(deprecated: library handles optimization)",
    )
    batch_size_cpu: int = Field(
        default=4,
        ge=1,
        le=16,
        description="CPU batch size (deprecated: library handles optimization)",
    )
    use_fp16: bool = Field(default=True, description="Enable FP16 acceleration")
    normalize_embeddings: bool = Field(
        default=True, description="L2 normalize embeddings"
    )
    return_dense: bool = Field(
        default=True, description="Return dense embeddings (1024D)"
    )
    return_sparse: bool = Field(default=True, description="Return sparse embeddings")
    return_colbert: bool = Field(
        default=False, description="Return ColBERT multi-vector embeddings"
    )
    device: str = Field(default="cuda", description="Target device (cuda/cpu/auto)")
    pooling_method: str = Field(
        default="cls", description="Pooling method ('cls', 'mean')"
    )
    weights_for_different_modes: list[float] = Field(
        default=[0.4, 0.2, 0.4],
        description="Weights for [dense, sparse, colbert] fusion",
    )
    return_numpy: bool = Field(
        default=False, description="Return numpy arrays instead of lists"
    )


class EmbeddingResult(BaseModel):
    """Result of BGE-M3 embedding operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dense_embeddings: list[list[float]] | None = Field(
        default=None, description="Dense embeddings (1024D)"
    )
    sparse_embeddings: list[dict[int, float]] | None = Field(
        default=None, description="Sparse embeddings"
    )
    colbert_embeddings: list[np.ndarray] | None = Field(
        default=None, description="ColBERT multi-vector embeddings"
    )
    processing_time: float = Field(description="Embedding processing time in seconds")
    batch_size: int = Field(description="Batch size used")
    memory_usage_mb: float = Field(description="Peak GPU memory usage in MB")
    model_info: dict[str, Any] = Field(
        default_factory=dict, description="Model information"
    )


class EmbeddingError(Exception):
    """Custom exception for embedding processing errors."""

    pass
