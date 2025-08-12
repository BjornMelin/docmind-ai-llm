"""Simplified configuration models for DocMind AI.

This module provides a streamlined configuration system using Pydantic BaseSettings
with .env file support. It consolidates all essential settings into a single class
and includes the AnalysisOutput model for structured document analysis results.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


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


# Rate limiting classes removed - not needed for local document processing app


class Settings(BaseSettings):
    """Essential configuration settings for DocMind AI.

    This class consolidates all core settings needed for the application,
    using Pydantic BaseSettings with .env file support for easy configuration.
    Only the most essential settings are included to minimize complexity.
    """

    # Core LLM Configuration
    llm_model: str = Field(default="gpt-4", env="DEFAULT_MODEL")
    embedding_model: str = Field(
        default="text-embedding-3-small", env="EMBEDDING_MODEL"
    )

    # Search and Retrieval
    similarity_top_k: int = Field(default=10)
    retrieval_top_k: int = Field(
        default=10, env="RETRIEVAL_TOP_K"
    )  # Alias for similarity_top_k
    hybrid_alpha: float = Field(default=0.7, env="RRF_FUSION_ALPHA")

    # Embedding processing
    embedding_batch_size: int = Field(default=100, env="EMBEDDING_BATCH_SIZE")

    # Hardware and Performance
    gpu_enabled: bool = Field(default=True, env="GPU_ACCELERATION")

    # Document Processing
    chunk_size: int = Field(default=1024)
    chunk_overlap: int = Field(default=200)
    parse_strategy: str = Field(
        default="hi_res",
        env="PARSE_STRATEGY",
        description=(
            "Document parsing strategy for Unstructured: 'auto' (default), "
            "'hi_res' (best quality), 'fast' (faster processing), "
            "'ocr_only' (OCR only), or 'vlm' (vision language model)"
        ),
    )

    # Reliability
    max_retries: int = Field(default=3)
    timeout: int = Field(default=30)

    # Optimization
    cache_enabled: bool = Field(default=True)

    # Infrastructure
    vector_store_type: str = Field(default="qdrant")
    rerank_enabled: bool = Field(default=True, env="ENABLE_COLBERT_RERANKING")

    # Rate limiting configuration removed - not needed for local app

    # Additional settings for backward compatibility with tests
    dense_embedding_dimension: int = Field(
        default=1024, env="DENSE_EMBEDDING_DIMENSION"
    )
    rrf_fusion_weight_dense: float = Field(default=0.7, env="RRF_FUSION_WEIGHT_DENSE")
    rrf_fusion_weight_sparse: float = Field(default=0.3, env="RRF_FUSION_WEIGHT_SPARSE")
    rrf_fusion_alpha: int = Field(default=60, env="RRF_FUSION_ALPHA")
    sparse_embedding_model: str | None = Field(
        default=None, env="SPARSE_EMBEDDING_MODEL"
    )
    dense_embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5", env="DENSE_EMBEDDING_MODEL"
    )
    enable_sparse_embeddings: bool = Field(
        default=False, env="ENABLE_SPARSE_EMBEDDINGS"
    )
    gpu_acceleration: bool = Field(
        default=True, env="GPU_ACCELERATION"
    )  # Alias for gpu_enabled
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")

    # Backend configuration settings
    ollama_base_url: str = Field(
        default="http://localhost:11434", env="OLLAMA_BASE_URL"
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1", env="LMSTUDIO_BASE_URL"
    )
    llamacpp_model_path: str = Field(
        default="/path/to/model.gguf", env="LLAMACPP_MODEL_PATH"
    )

    # Additional fields for test compatibility
    reranker_model: str = Field(
        default="jinaai/jina-reranker-v2-base-multilingual", env="RERANKER_MODEL"
    )
    reranking_top_k: int = Field(default=5, env="RERANKING_TOP_K")
    default_model: str = Field(default="gpt-4", env="DEFAULT_MODEL")

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("rrf_fusion_weight_dense", "rrf_fusion_weight_sparse")
    @classmethod
    def validate_rrf_weight_range(cls, v: float) -> float:
        """Validate RRF weights are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("RRF weight must be between 0 and 1")
        return v

    @field_validator("rrf_fusion_weight_sparse")
    @classmethod
    def validate_rrf_weights_sum(cls, v: float, info) -> float:
        """Validate that RRF weights sum to 1.0 (within tolerance)."""
        if "rrf_fusion_weight_dense" in info.data:
            dense_weight = info.data["rrf_fusion_weight_dense"]
            total = dense_weight + v
            if abs(total - 1.0) > 0.001:  # Tolerance for floating point
                raise ValueError("RRF weights must sum to 1.0")
        return v

    @field_validator("dense_embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, v: int) -> int:
        """Validate embedding dimension is positive and reasonable."""
        if v <= 0:
            raise ValueError("Embedding dimension must be positive")
        if v > 10000:
            raise ValueError("Embedding dimension seems too large")
        return v

    @field_validator("dense_embedding_model")
    @classmethod
    def validate_bge_model_dimension(cls, v: str, info) -> str:
        """Validate BGE-Large model has correct dimension."""
        if "bge-large-en" in v.lower() and "dense_embedding_dimension" in info.data:
            dimension = info.data["dense_embedding_dimension"]
            if dimension != 1024:
                raise ValueError("BGE-Large model requires 1024 dimensions")
        return v

    @field_validator("sparse_embedding_model")
    @classmethod
    def validate_splade_model(cls, v: str | None) -> str | None:
        """Validate SPLADE++ model name."""
        if (
            v
            and v.lower() not in [None, "none", ""]
            and "splade" in v.lower()
            and "prithivida/Splade_PP_en_v1" not in v
        ):
            raise ValueError("Invalid SPLADE++ model name")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Validate chunk overlap is smaller than chunk size."""
        if "chunk_size" in info.data:
            chunk_size = info.data["chunk_size"]
            if v >= chunk_size:
                raise ValueError(
                    f"Chunk size ({chunk_size}) must be larger than overlap ({v})"
                )
        return v

    @field_validator("parse_strategy")
    @classmethod
    def validate_parse_strategy(cls, v: str) -> str:
        """Validate parse strategy is one of the supported options."""
        valid_strategies = {"auto", "hi_res", "fast", "ocr_only", "vlm"}
        if v not in valid_strategies:
            raise ValueError(
                f"Parse strategy must be one of {valid_strategies}, got '{v}'"
            )
        return v

    # get_rate_limit_config method removed - not needed for local app


# Create global settings instance
settings = Settings()

# For backward compatibility, also expose as AppSettings
AppSettings = Settings

__all__ = [
    "Settings",
    "AppSettings",
    "AnalysisOutput",
    "settings",
]
