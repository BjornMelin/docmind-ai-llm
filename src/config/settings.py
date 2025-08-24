"""Configuration settings for DocMind AI with Multi-Agent Coordination.

This module provides centralized configuration management for all application
components including multi-agent settings, LLM backends, and performance tuning.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnalysisOutput(BaseModel):
    """Structured output schema for document analysis results.

    Defines the expected format for analysis results from the language model,
    ensuring consistent structure for summaries, insights, action items, and
    questions. Used with Pydantic output parsing to validate and structure
    LLM responses.
    """

    summary: str = Field(description="Summary of the document")
    key_insights: list[str] = Field(description="Key insights extracted")
    action_items: list[str] = Field(description="Action items identified")
    open_questions: list[str] = Field(description="Open questions surfaced")


class Settings(BaseSettings):
    """Application settings with multi-agent configuration support."""

    # Application metadata
    app_name: str = "DocMind AI"
    app_version: str = "2.0.0"
    debug: bool = Field(default=False, description="Enable debug mode")

    # Multi-Agent Coordination Settings (REQ-0001 to REQ-0010)
    enable_multi_agent: bool = Field(
        default=True,
        description="Enable multi-agent coordination system",
    )
    agent_decision_timeout: int = Field(
        default=300,
        description="Agent decision timeout in milliseconds (REQ-0007)",
        ge=100,
        le=1000,
    )
    enable_fallback_rag: bool = Field(
        default=True,
        description="Enable fallback to basic RAG on agent failure (REQ-0008)",
    )
    max_agent_retries: int = Field(
        default=2,
        description="Maximum retries for agent operations",
        ge=0,
        le=5,
    )

    # LLM Backend Configuration (REQ-0009: Local execution only)
    llm_backend: Literal["ollama", "llamacpp", "vllm", "openai"] = Field(
        default="vllm",
        description=(
            "LLM backend to use (local-first, vLLM default for FP8 optimization)"
        ),
    )
    model_name: str = Field(
        default="Qwen/Qwen3-4B-Instruct-2507",
        description="Model name for LLM (REQ-0063-v2: Qwen3-4B-Instruct-2507 with AWQ)",
    )
    llm_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for LLM backend",
    )
    llm_api_key: str | None = Field(
        default=None,
        description="API key if required (for OpenAI compatibility)",
    )
    llm_temperature: float = Field(
        default=0.1,
        description="Temperature for LLM generation",
        ge=0.0,
        le=2.0,
    )
    llm_max_tokens: int = Field(
        default=2048,
        description="Maximum tokens for generation",
        ge=128,
        le=8192,
    )

    # Model Optimization Settings (REQ-0063-v2, REQ-0064-v2)
    quantization: str = Field(
        default="fp8",
        description="Quantization method for model (FP8 for weights and activations)",
    )
    kv_cache_dtype: str = Field(
        default="fp8",
        description=(
            "KV cache data type for memory optimization (FP8 for maximum efficiency)"
        ),
    )
    enable_kv_cache_optimization: bool = Field(
        default=True,
        description="Enable KV cache optimization for performance",
    )
    kv_cache_performance_boost: float = Field(
        default=1.3,
        description="Expected performance boost from KV cache optimization (30%)",
        ge=1.0,
        le=2.0,
    )

    # Context Management (REQ-0094-v2: Expanded context)
    context_window_size: int = Field(
        default=131072,
        description=(
            "Context window size in tokens (128K with FP8 optimization for vLLM)"
        ),
    )
    context_buffer_size: int = Field(
        default=131072,
        description="Maximum context buffer size (128K tokens with FP8)",
    )
    enable_conversation_memory: bool = Field(
        default=True,
        description="Enable conversation memory across interactions",
    )

    # Document Processing Configuration
    chunk_size: int = Field(
        default=512,
        description="Text chunk size for processing",
        ge=128,
        le=2048,
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks",
        ge=0,
        le=200,
    )
    enable_document_caching: bool = Field(
        default=True,
        description="Enable document parsing cache",
    )
    max_document_size_mb: int = Field(
        default=100,
        description="Maximum document size in MB",
        ge=1,
        le=500,
    )
    # Additional compatibility fields from models/core.py
    parse_strategy: str = Field(
        default="hi_res",
        description="Document parsing strategy for Unstructured",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for operations",
        ge=0,
        le=10,
    )
    timeout: int = Field(
        default=30,
        description="Operation timeout in seconds",
        ge=1,
        le=300,
    )

    # Retrieval Configuration
    retrieval_strategy: Literal["vector", "hybrid", "graphrag"] = Field(
        default="hybrid",
        description="Default retrieval strategy",
    )
    top_k: int = Field(
        default=10,
        description="Number of documents to retrieve",
        ge=1,
        le=50,
    )
    use_reranking: bool = Field(
        default=True,
        description="Enable BGE reranking (REQ-0045)",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Reranking model to use",
    )
    reranking_top_k: int = Field(
        default=5,
        description="Top-k after reranking",
        ge=1,
        le=20,
    )

    # Embedding Configuration
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Dense embedding model (REQ-0042)",
    )
    embedding_dimension: int = Field(
        default=1024,
        description="Embedding dimension",
    )
    use_sparse_embeddings: bool = Field(
        default=True,
        description="Enable SPLADE++ sparse embeddings (REQ-0043)",
    )
    sparse_embedding_model: str | None = Field(
        default=None,
        description="Sparse embedding model (SPLADE++)",
    )
    embedding_batch_size: int = Field(
        default=100,
        description="Batch size for embedding processing",
        ge=1,
        le=1000,
    )

    # Vector Database Configuration (REQ-0047)
    vector_store_type: Literal["qdrant", "chroma", "weaviate"] = Field(
        default="qdrant",
        description="Vector database type",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    qdrant_collection: str = Field(
        default="docmind_docs",
        description="Qdrant collection name",
    )

    # DSPy Optimization (REQ-0091)
    enable_dspy_optimization: bool = Field(
        default=True,
        description="Enable DSPy prompt optimization",
    )
    dspy_optimization_samples: int = Field(
        default=20,
        description="Number of samples for DSPy optimization",
        ge=5,
        le=100,
    )

    # Performance Configuration (REQ-0007, REQ-0046)
    max_query_latency_ms: int = Field(
        default=2000,
        description="Maximum query latency in milliseconds",
    )
    max_memory_gb: float = Field(
        default=4.0,
        description="Maximum RAM usage in GB (REQ-0069)",
    )
    max_vram_gb: float = Field(
        default=14.0,
        description=(
            "Maximum VRAM usage in GB (REQ-0070, optimized for FP8 quantization)"
        ),
    )
    enable_gpu_acceleration: bool = Field(
        default=True,
        description="Enable GPU acceleration if available",
    )
    # Backend configuration compatibility
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL",
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        description="LMStudio server base URL",
    )
    llamacpp_model_path: str = Field(
        default="/path/to/model.gguf",
        description="llama.cpp model file path",
    )

    # vLLM-Specific Settings (REQ-0063-v2, REQ-0064-v2)
    vllm_gpu_memory_utilization: float = Field(
        default=0.85,
        description="GPU memory utilization for vLLM (conservative for 16GB VRAM)",
        ge=0.1,
        le=0.95,
    )
    vllm_attention_backend: str = Field(
        default="FLASHINFER",
        description="Attention backend for vLLM (FlashInfer for optimization)",
    )
    vllm_enable_chunked_prefill: bool = Field(
        default=True,
        description="Enable chunked prefill for memory efficiency",
    )
    vllm_max_num_batched_tokens: int = Field(
        default=8192,
        description="Maximum batched tokens for vLLM prefill optimization",
        ge=1024,
        le=16384,
    )
    vllm_max_num_seqs: int = Field(
        default=16,
        description="Maximum number of sequences in vLLM batch",
        ge=1,
        le=64,
    )

    # Persistence Configuration
    data_dir: Path = Field(
        default=Path("./data"),
        description="Directory for persistent data",
    )
    cache_dir: Path = Field(
        default=Path("./cache"),
        description="Directory for cache files",
    )
    sqlite_db_path: Path = Field(
        default=Path("./data/docmind.db"),
        description="SQLite database path",
    )
    enable_wal_mode: bool = Field(
        default=True,
        description="Enable SQLite WAL mode (REQ-0067)",
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_file: Path | None = Field(
        default=Path("./logs/docmind.log"),
        description="Log file path",
    )
    enable_performance_logging: bool = Field(
        default=True,
        description="Enable performance metrics logging",
    )

    # UI Configuration
    streamlit_port: int = Field(
        default=8501,
        description="Streamlit UI port",
        ge=1024,
        le=65535,
    )
    enable_ui_dark_mode: bool = Field(
        default=True,
        description="Enable dark mode by default",
    )

    # Advanced Features
    enable_graphrag: bool = Field(
        default=False,
        description="Enable GraphRAG for relationship queries (REQ-0049)",
    )
    enable_multimodal: bool = Field(
        default=False,
        description="Enable multimodal processing (REQ-0044)",
    )
    analysis_modes: list[str] = Field(
        default=["detailed", "summary", "comparison"],
        description="Available analysis modes (REQ-0095)",
    )

    # Centralized Constants - Eliminating DRY violations from recent remediation

    # Memory and Size Conversion Constants
    bytes_to_gb_divisor: int = Field(
        default=1024**3,
        description="Divisor to convert bytes to GB (1024^3)",
    )
    bytes_to_mb_divisor: int = Field(
        default=1024 * 1024,
        description="Divisor to convert bytes to MB (1024^2)",
    )

    # BGE-M3 Model Constants
    bge_m3_embedding_dim: int = Field(
        default=1024,
        description="BGE-M3 embedding dimension",
        ge=512,
        le=4096,
    )
    bge_m3_max_length: int = Field(
        default=8192,
        description="BGE-M3 maximum token length",
        ge=512,
        le=16384,
    )
    bge_m3_model_name: str = Field(
        default="BAAI/bge-m3",
        description="BGE-M3 model identifier",
    )
    bge_m3_batch_size_gpu: int = Field(
        default=12,
        description="BGE-M3 GPU batch size",
        ge=1,
        le=128,
    )
    bge_m3_batch_size_cpu: int = Field(
        default=4,
        description="BGE-M3 CPU batch size",
        ge=1,
        le=32,
    )

    # Hybrid Retrieval Constants
    rrf_k_constant: int = Field(
        default=60,
        description="RRF constant for reciprocal rank calculation",
        ge=10,
        le=100,
    )
    # Additional RRF fusion compatibility fields
    rrf_fusion_weight_dense: float = Field(
        default=0.7,
        description="RRF fusion weight for dense embeddings",
        ge=0.0,
        le=1.0,
    )
    rrf_fusion_weight_sparse: float = Field(
        default=0.3,
        description="RRF fusion weight for sparse embeddings",
        ge=0.0,
        le=1.0,
    )
    rrf_fusion_alpha: int = Field(
        default=60,
        description="RRF fusion alpha parameter",
        ge=10,
        le=100,
    )

    # Default Processing Values
    default_batch_size: int = Field(
        default=20,
        description="Standard batch size for operations",
        ge=1,
        le=128,
    )
    default_confidence_threshold: float = Field(
        default=0.8,
        description="Default confidence threshold for operations",
        ge=0.0,
        le=1.0,
    )
    default_entity_confidence: float = Field(
        default=0.8,
        description="Default entity extraction confidence",
        ge=0.0,
        le=1.0,
    )

    # Query Engine Constants

    # Timeout Configuration
    default_qdrant_timeout: int = Field(
        default=60,
        description="Default Qdrant operation timeout in seconds",
        ge=10,
        le=300,
    )
    default_agent_timeout: float = Field(
        default=3.0,
        description="Default agent operation timeout in seconds",
        ge=1.0,
        le=10.0,
    )

    # Cache and Processing
    cache_expiry_seconds: int = Field(
        default=3600,
        description="Cache expiry time in seconds (1 hour)",
        ge=300,
        le=86400,
    )
    spacy_download_timeout: int = Field(
        default=300,
        description="spaCy model download timeout in seconds",
        ge=60,
        le=1200,
    )

    # Performance Monitoring
    cpu_monitoring_interval: float = Field(
        default=0.1,
        description="CPU monitoring interval in seconds",
        ge=0.01,
        le=1.0,
    )
    percent_multiplier: int = Field(
        default=100,
        description="Multiplier for percentage calculations",
    )

    # App.py Constants - moved to eliminate duplication
    default_token_limit: int = Field(
        default=131072,
        description="Default token limit (128K context)",
        ge=1024,
        le=1000000,
    )
    context_size_options: list[int] = Field(
        default=[8192, 32768, 65536, 131072],
        description="Available context size options",
    )
    suggested_context_high: int = Field(
        default=65536,
        description="Suggested high context size (64K)",
        ge=8192,
        le=131072,
    )
    suggested_context_medium: int = Field(
        default=32768,
        description="Suggested medium context size (32K)",
        ge=8192,
        le=131072,
    )
    suggested_context_low: int = Field(
        default=8192,
        description="Suggested low context size (8K)",
        ge=1024,
        le=131072,
    )
    request_timeout_seconds: float = Field(
        default=60.0,
        description="Request timeout in seconds",
        ge=1.0,
        le=300.0,
    )
    streaming_delay_seconds: float = Field(
        default=0.02,
        description="Streaming delay between chunks in seconds",
        ge=0.001,
        le=1.0,
    )
    minimum_vram_high_gb: int = Field(
        default=16,
        description="Minimum VRAM for high performance (GB)",
        ge=4,
        le=80,
    )
    minimum_vram_medium_gb: int = Field(
        default=8,
        description="Minimum VRAM for medium performance (GB)",
        ge=2,
        le=32,
    )

    @field_validator(
        "data_dir", "cache_dir", "log_file", "sqlite_db_path", mode="before"
    )
    @classmethod
    def create_directories(cls, v: Path | str | None) -> Path | None:
        """Create directories if they don't exist."""
        if v is not None:
            path = Path(v)
            if path.suffix:  # It's a file
                path.parent.mkdir(parents=True, exist_ok=True)
            else:  # It's a directory
                path.mkdir(parents=True, exist_ok=True)
            return path
        return None

    @field_validator("app_name", "model_name", "embedding_model", "bge_m3_model_name")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        """Validate that critical string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace-only")
        return v.strip()

    @field_validator("llm_backend")
    @classmethod
    def validate_llm_backend(cls, v: str) -> str:
        """Validate LLM backend selection."""
        if v == "openai":
            # Note: OpenAI is allowed but will use local endpoints
            import warnings

            warnings.warn(
                "OpenAI backend selected - ensure using local-compatible endpoint",
                UserWarning,
                stacklevel=2,
            )
        return v

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

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, v: int) -> int:
        """Validate embedding dimension is positive and reasonable."""
        if v <= 0:
            raise ValueError("Embedding dimension must be positive")
        if v > 10000:
            raise ValueError("Embedding dimension seems too large")
        return v

    @field_validator("embedding_model")
    @classmethod
    def validate_bge_model_dimension(cls, v: str, info) -> str:
        """Validate BGE-Large model has correct dimension."""
        if "bge-large-en" in v.lower():
            if "embedding_dimension" in info.data:
                dimension = info.data["embedding_dimension"]
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="DOCMIND_",
        extra="ignore",  # Ignore extra environment variables
        validate_default=True,  # Validate default values
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary."""
        return self.model_dump()

    def get_agent_config(self) -> dict[str, Any]:
        """Get configuration specific to multi-agent system."""
        return {
            "enable_multi_agent": self.enable_multi_agent,
            "agent_decision_timeout": self.agent_decision_timeout,
            "enable_fallback_rag": self.enable_fallback_rag,
            "max_agent_retries": self.max_agent_retries,
            "llm_backend": self.llm_backend,
            "model_name": self.model_name,
            "context_window_size": self.context_window_size,
            "context_buffer_size": self.context_buffer_size,
            "quantization": self.quantization,
            "kv_cache_dtype": self.kv_cache_dtype,
        }

    def get_performance_config(self) -> dict[str, Any]:
        """Get performance-related configuration."""
        return {
            "max_query_latency_ms": self.max_query_latency_ms,
            "agent_decision_timeout": self.agent_decision_timeout,
            "max_memory_gb": self.max_memory_gb,
            "max_vram_gb": self.max_vram_gb,
            "enable_gpu_acceleration": self.enable_gpu_acceleration,
            "enable_performance_logging": self.enable_performance_logging,
            "vllm_gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "vllm_attention_backend": self.vllm_attention_backend,
            "vllm_enable_chunked_prefill": self.vllm_enable_chunked_prefill,
            "vllm_max_num_batched_tokens": self.vllm_max_num_batched_tokens,
            "vllm_max_num_seqs": self.vllm_max_num_seqs,
        }

    def get_vllm_config(self) -> dict[str, Any]:
        """Get vLLM-specific configuration."""
        return {
            "model_name": self.model_name,
            "quantization": self.quantization,
            "kv_cache_dtype": self.kv_cache_dtype,
            "max_model_len": self.context_window_size,
            "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "attention_backend": self.vllm_attention_backend,
            "enable_chunked_prefill": self.vllm_enable_chunked_prefill,
            "max_num_batched_tokens": self.vllm_max_num_batched_tokens,
            "max_num_seqs": self.vllm_max_num_seqs,
            "default_temperature": self.llm_temperature,
            "default_max_tokens": self.llm_max_tokens,
        }


# Global settings instance
settings = Settings()

# For backward compatibility, also expose as AppSettings
AppSettings = Settings

__all__ = [
    "Settings",
    "AppSettings",
    "AnalysisOutput",
    "settings",
]
