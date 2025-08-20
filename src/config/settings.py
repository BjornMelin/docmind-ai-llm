"""Configuration settings for DocMind AI with Multi-Agent Coordination.

This module provides centralized configuration management for all application
components including multi-agent settings, LLM backends, and performance tuning.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        description="LLM backend to use (local-first, vLLM default for FP8 optimization)",
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
        description="KV cache data type for memory optimization (FP8 for maximum efficiency)",
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
        description="Context window size in tokens (128K with FP8 optimization for vLLM)",
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
    reranker_top_k: int = Field(
        default=5,
        description="Top K after reranking",
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
        description="Maximum VRAM usage in GB (REQ-0070, optimized for FP8 quantization)",
    )
    enable_gpu_acceleration: bool = Field(
        default=True,
        description="Enable GPU acceleration if available",
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

    @field_validator("data_dir", "cache_dir", "log_file", mode="before")
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
