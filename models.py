"""Pydantic models for DocMind AI.

This module defines the data models and configuration schemas used throughout
the DocMind AI application. It includes structured output models for document
analysis results and application settings loaded from environment variables.

The models ensure type safety, data validation, and provide clear interfaces
for the application's core data structures.

Example:
    Using the models::

        from models import AnalysisOutput, AppSettings

        # Create structured analysis output
        result = AnalysisOutput(
            summary="Document summary",
            key_insights=["insight1", "insight2"],
            action_items=["action1"],
            open_questions=["question1"]
        )

        # Load application settings
        settings = AppSettings()
        print(settings.ollama_base_url)

Classes:
    AnalysisOutput: Structured schema for document analysis results.
    AppSettings: Application configuration loaded from environment variables.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnalysisOutput(BaseModel):
    """Structured output schema for document analysis results.

    Defines the expected format for analysis results from the language model,
    ensuring consistent structure for summaries, insights, action items, and
    questions. Used with Pydantic output parsing to validate and structure
    LLM responses.

    Attributes:
        summary: Brief overview of the document content and main themes.
        key_insights: List of important findings and observations extracted
            from the document.
        action_items: List of concrete steps or tasks identified from the
            document content.
        open_questions: List of unresolved questions or areas requiring
            further investigation.

    """

    summary: str = Field(description="Summary of the document")
    key_insights: list[str] = Field(description="Key insights extracted")
    action_items: list[str] = Field(description="Action items identified")
    open_questions: list[str] = Field(description="Open questions surfaced")


class AppSettings(BaseSettings):
    """Enhanced application configuration settings with advanced embedding support.

    Manages all configurable parameters for the DocMind AI application,
    including backend URLs, model specifications, service endpoints, and advanced
    hybrid search configurations. Settings are automatically loaded from
    environment variables or .env file.

    Advanced Features:
        - Research-backed BGE-Large dense embeddings (BAAI/bge-large-en-v1.5)
        - SPLADE++ sparse embeddings for hybrid search (prithvida/Splade_PP_en_v1)
        - RRF fusion parameters optimized from research (0.7/0.3 weight distribution)
        - GPU acceleration toggles and batch processing optimization
        - ColBERT reranking pipeline configuration
        - Unstructured library for multimodal document parsing

    Attributes:
        backend: Default backend type for LLM inference.
        ollama_base_url: Base URL for Ollama server API.
        lmstudio_base_url: Base URL for LM Studio server API.
        llamacpp_model_path: File path to the Llama.cpp model file.
        default_model: Default model name/identifier to use.
        context_size: Maximum context window size for language models.
        qdrant_url: URL for the Qdrant vector database server.

        # Dense Embedding Configuration (Research-backed BGE-Large)
        dense_embedding_model: Primary dense embedding model for semantic search.
        dense_embedding_dimension: Vector dimension for dense embeddings.

        # Sparse Embedding Configuration (Research-backed SPLADE++)
        sparse_embedding_model: Model for sparse embeddings in hybrid search.
        enable_sparse_embeddings: Toggle for sparse embedding computation.

        # RRF Fusion Parameters (Research-optimized weights)
        rrf_fusion_weight_dense: Weight for dense embeddings in RRF fusion.
        rrf_fusion_weight_sparse: Weight for sparse embeddings in RRF fusion.
        rrf_fusion_alpha: Alpha parameter for RRF fusion algorithm.

        # GPU Acceleration Configuration
        gpu_acceleration: Enable GPU acceleration for embeddings and search.
        cuda_device_id: CUDA device ID for GPU operations.
        embedding_batch_size: Batch size for embedding computation.
        prefetch_factor: Prefetch factor for DataLoader optimization.

        # Performance Optimization
        enable_quantization: Enable scalar quantization for memory optimization.
        quantization_type: Type of quantization (int8, int4).
        max_concurrent_requests: Maximum concurrent embedding requests.

        # Reranking Configuration
        default_reranker_model: Model name for document reranking.
        enable_colbert_reranking: Enable ColBERT late interaction reranking.
        reranking_top_k: Number of documents to rerank.

        # Document Processing Configuration
        parse_strategy: Parsing strategy for Unstructured library.
        chunk_size: Optimal chunk size for embeddings.
        chunk_overlap: Overlap between chunks for context preservation.
        max_entities: Maximum entities for knowledge graph extraction.

        model_config: Pydantic configuration for settings loading.

    """

    # Core Backend Configuration
    backend: str = Field(
        default="ollama", description="Default backend type for LLM inference"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Base URL for Ollama server API"
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        description="Base URL for LM Studio server API",
    )
    llamacpp_model_path: str = Field(
        default="/path/to/model.gguf",
        description="File path to the Llama.cpp model file",
    )
    default_model: str = Field(
        default="google/gemma-3n-E4B-it",
        description="Default base model with hardware-adaptive variants",
    )
    context_size: int = Field(
        default=4096,
        ge=1,
        description="Maximum context window size for language models",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="URL for the Qdrant vector database server",
    )
    qdrant_pool_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of connections in Qdrant connection pool",
    )

    # Dense Embedding Configuration (Research-backed BGE-Large)
    dense_embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Research-backed BGE-Large model for optimal dense embeddings",
    )
    dense_embedding_dimension: int = Field(
        default=1024, ge=1, description="Vector dimension for BGE-Large embeddings"
    )

    @field_validator("dense_embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, v: int) -> int:
        """Validate embedding dimension is positive and reasonable.

        Args:
            v: Embedding dimension to validate.

        Returns:
            Validated dimension.

        Raises:
            ValueError: If dimension is invalid.
        """
        if v <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {v}")
        if v > 10000:
            raise ValueError(f"Embedding dimension seems too large: {v}")
        return v

    # Sparse Embedding Configuration (Research-backed SPLADE++)
    sparse_embedding_model: str = Field(
        default="prithivida/Splade_PP_en_v1",
        description="Research-backed SPLADE++ model for sparse embeddings",
    )
    enable_sparse_embeddings: bool = Field(
        default=True, description="Enable sparse embeddings for hybrid search"
    )

    # RRF Fusion Parameters (Research-optimized)
    rrf_fusion_weight_dense: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Research-optimized weight for dense embeddings in RRF fusion",
    )
    rrf_fusion_weight_sparse: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Research-optimized weight for sparse embeddings in RRF fusion",
    )
    rrf_fusion_alpha: int = Field(
        default=60,
        ge=1,
        description="Alpha parameter for RRF fusion algorithm (from Qdrant research)",
    )

    @field_validator("rrf_fusion_weight_dense", "rrf_fusion_weight_sparse")
    @classmethod
    def validate_weight_range(cls, v: float) -> float:
        """Validate RRF weights are in valid range [0, 1].

        Args:
            v: Weight value to validate.

        Returns:
            Validated weight value.

        Raises:
            ValueError: If weight is not in range [0, 1].
        """
        if not 0 <= v <= 1:
            raise ValueError(f"RRF weight must be between 0 and 1, got {v}")
        return v

    @model_validator(mode="after")
    def validate_rrf_weights_sum(self) -> "AppSettings":
        """Validate that RRF weights sum to 1.0.

        Returns:
            Validated settings instance.

        Raises:
            ValueError: If weights don't sum to 1.0 (within tolerance).
        """
        total = self.rrf_fusion_weight_dense + self.rrf_fusion_weight_sparse
        if abs(total - 1.0) > 0.001:  # Small tolerance for floating point
            raise ValueError(
                f"RRF weights must sum to 1.0, got "
                f"dense={self.rrf_fusion_weight_dense}, "
                f"sparse={self.rrf_fusion_weight_sparse}, total={total:.3f}"
            )
        return self

    # GPU Acceleration Configuration
    gpu_acceleration: bool = Field(
        default=True,
        description="Enable GPU acceleration for 100x performance improvement",
    )
    cuda_device_id: int = Field(
        default=0, ge=0, description="CUDA device ID for GPU operations"
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size for embedding computation (GPU-optimized)",
    )
    prefetch_factor: int = Field(
        default=2, ge=1, le=8, description="Prefetch factor for DataLoader optimization"
    )

    # Performance Optimization Configuration
    enable_quantization: bool = Field(
        default=True, description="Enable scalar quantization for 4x memory reduction"
    )
    quantization_type: str = Field(
        default="int8", description="Quantization type for memory optimization"
    )
    max_concurrent_requests: int = Field(
        default=10, ge=1, le=100, description="Maximum concurrent embedding requests"
    )

    # Reranking Configuration
    default_reranker_model: str = Field(
        default="jinaai/jina-reranker-v2-base-multilingual",
        description="Default reranker model for document ranking",
    )
    enable_colbert_reranking: bool = Field(
        default=True, description="Enable ColBERT late interaction reranking"
    )
    reranking_top_k: int = Field(
        default=5,
        ge=5,
        le=100,
        description=(
            "Number of documents to rerank (Phase 2.2: retrieve 20, rerank to 5)"
        ),
    )

    # Document Processing Configuration (Unstructured)
    parse_strategy: str = Field(
        default="hi_res",
        description="Parsing strategy: 'hi_res' for best quality, 'fast' for speed",
    )
    chunk_size: int = Field(
        default=1024,
        ge=256,
        le=4096,
        description="Optimal chunk size for embeddings (1024 recommended)",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=512,
        description="Overlap between chunks for context preservation",
    )
    max_entities: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum entities for knowledge graph extraction",
    )

    # Debug and Development Configuration
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode with profiling and additional logging",
    )

    # Query Configuration
    similarity_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top similar documents to retrieve",
    )

    # Compatibility alias for reranker
    reranker_model: str | None = Field(
        default=None,
        description="Reranker model name (alias for default_reranker_model)",
    )

    def model_post_init(self, __context: Any) -> None:
        """Set up compatibility aliases after model initialization."""
        if self.reranker_model is None:
            object.__setattr__(self, "reranker_model", self.default_reranker_model)

    # Legacy Support (backward compatibility)
    default_embedding_model: str | None = Field(
        default="jinaai/jina-embeddings-v4",
        description="Legacy embedding model (deprecated, use dense_embedding_model)",
        alias="DEFAULT_EMBEDDING_MODEL",  # For env var compatibility
    )

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> "AppSettings":
        """Validate chunk configuration is consistent.

        Returns:
            Validated settings instance.

        Raises:
            ValueError: If chunk configuration is invalid.
        """
        if self.chunk_size <= self.chunk_overlap:
            raise ValueError(
                f"Chunk size ({self.chunk_size}) must be larger than "
                f"chunk overlap ({self.chunk_overlap})"
            )
        return self

    @model_validator(mode="after")
    def validate_model_compatibility(self) -> "AppSettings":
        """Validate that model configurations are compatible.

        Returns:
            Validated settings instance.

        Raises:
            ValueError: If model settings are incompatible.
        """
        # Validate BGE-Large dimension
        if (
            self.dense_embedding_model == "BAAI/bge-large-en-v1.5"
            and self.dense_embedding_dimension != 1024
        ):
            raise ValueError(
                f"BGE-Large model requires 1024 dimensions, "
                f"got {self.dense_embedding_dimension}"
            )

        # Validate SPLADE++ model name
        if (
            self.sparse_embedding_model
            and "prithivida" not in self.sparse_embedding_model
        ):
            raise ValueError(
                f"Invalid SPLADE++ model name: {self.sparse_embedding_model}. "
                'Expected "prithivida/Splade_PP_en_v1"'
            )

        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        case_sensitive=False,
        env_prefix="",  # No prefix for env vars
        extra="ignore",  # Ignore extra env vars not in model
    )
