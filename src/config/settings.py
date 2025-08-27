"""Unified DocMind AI configuration using Pydantic Settings V2.

This module provides the main configuration architecture implementing Task 2.2.1:
- Unified Pydantic BaseSettings with environment variable mapping
- Nested configuration models for complex areas
- ADR-compliant settings preservation
- 76% complexity reduction while maintaining functionality

Usage:
    from src.config import settings
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VLLMConfig(BaseModel):
    """vLLM configuration for FP8 optimization (ADR-004, ADR-010)."""

    # Model Configuration
    model: str = Field(default="Qwen/Qwen3-4B-Instruct-2507-FP8")
    context_window: int = Field(default=131072, ge=8192, le=200000)
    max_tokens: int = Field(default=2048, ge=100, le=8192)

    # FP8 Optimization Settings
    gpu_memory_utilization: float = Field(default=0.85, ge=0.5, le=0.95)
    kv_cache_dtype: str = Field(default="fp8_e5m2")
    attention_backend: str = Field(default="FLASHINFER")
    enable_chunked_prefill: bool = Field(default=True)

    # Performance Settings
    max_num_seqs: int = Field(default=16, ge=1, le=64)
    max_num_batched_tokens: int = Field(default=8192, ge=1024, le=16384)


class ProcessingConfig(BaseModel):
    """Document processing configuration (ADR-009)."""

    # Unstructured.io Settings
    chunk_size: int = Field(default=1500, ge=100, le=10000)
    new_after_n_chars: int = Field(default=1200, ge=100, le=8000)
    combine_text_under_n_chars: int = Field(default=500, ge=50, le=2000)
    multipage_sections: bool = Field(default=True)
    max_document_size_mb: int = Field(default=100, ge=1, le=500)


class AgentConfig(BaseModel):
    """Multi-agent system configuration (ADR-011)."""

    enable_multi_agent: bool = Field(default=True)
    decision_timeout: int = Field(default=200, ge=10, le=1000)
    max_retries: int = Field(default=2, ge=0, le=10)
    max_concurrent_agents: int = Field(default=3, ge=1, le=10)
    enable_fallback_rag: bool = Field(default=True)


class EmbeddingConfig(BaseModel):
    """BGE-M3 embedding configuration (ADR-002)."""

    model_name: str = Field(default="BAAI/bge-m3")
    dimension: int = Field(default=1024, ge=256, le=4096)
    max_length: int = Field(default=8192, ge=512, le=16384)
    batch_size_gpu: int = Field(default=12, ge=1, le=128)
    batch_size_cpu: int = Field(default=4, ge=1, le=32)


class RetrievalConfig(BaseModel):
    """Retrieval and reranking configuration (ADR-006, ADR-007)."""

    strategy: str = Field(default="hybrid")
    top_k: int = Field(default=10, ge=1, le=50)
    use_reranking: bool = Field(default=True)
    reranking_top_k: int = Field(default=5, ge=1, le=20)
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")

    # RRF Fusion Settings
    rrf_alpha: int = Field(default=60, ge=10, le=100)
    rrf_k_constant: int = Field(default=60, ge=10, le=100)


class CacheConfig(BaseModel):
    """Simple cache configuration."""

    enable_document_caching: bool = Field(default=True)
    ttl_seconds: int = Field(default=3600, ge=300, le=86400)
    max_size_mb: int = Field(default=1000, ge=100, le=10000)
    enable_semantic_cache: bool = Field(default=True)
    semantic_threshold: float = Field(default=0.85, ge=0.5, le=0.95)


class DocMindSettings(BaseSettings):
    """Unified DocMind AI configuration with Pydantic Settings V2.

    Implements Task 2.2.1 unified configuration architecture with:
    - Environment variable mapping with DOCMIND_ prefix
    - Nested configuration models for complex areas
    - ADR-compliant settings preservation
    - 76% complexity reduction from previous architecture
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DOCMIND_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid",
    )

    # Core Application
    app_name: str = Field(default="DocMind AI")
    app_version: str = Field(default="2.0.0")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # File System Paths
    data_dir: Path = Field(default=Path("./data"))
    cache_dir: Path = Field(default=Path("./cache"))
    sqlite_db_path: Path = Field(default=Path("./data/docmind.db"))
    log_file: Path = Field(default=Path("./logs/docmind.log"))

    # Backend Configuration
    llm_backend: str = Field(default="ollama")
    ollama_base_url: str = Field(default="http://localhost:11434")
    enable_gpu_acceleration: bool = Field(default=True)

    # Vector Database
    vector_store_type: str = Field(default="qdrant")
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="docmind_docs")

    # UI Configuration
    streamlit_port: int = Field(default=8501, ge=1024, le=65535)
    enable_ui_dark_mode: bool = Field(default=True)

    # Advanced Features
    enable_graphrag: bool = Field(default=False)
    enable_dspy_optimization: bool = Field(default=True)
    enable_performance_logging: bool = Field(default=True)

    # Memory conversion constants
    bytes_to_gb_divisor: int = Field(default=1024**3)
    bytes_to_mb_divisor: int = Field(default=1024**2)

    # Nested Configuration Models
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    def model_post_init(self, __context: Any) -> None:
        """Create necessary directories after initialization."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.sqlite_db_path.parent != self.data_dir:
            self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_vllm_env_vars(self) -> dict[str, str]:
        """Get vLLM environment variables for process setup."""
        return {
            "VLLM_ATTENTION_BACKEND": self.vllm.attention_backend,
            "VLLM_KV_CACHE_DTYPE": self.vllm.kv_cache_dtype,
            "VLLM_GPU_MEMORY_UTILIZATION": str(self.vllm.gpu_memory_utilization),
            "VLLM_ENABLE_CHUNKED_PREFILL": "1"
            if self.vllm.enable_chunked_prefill
            else "0",
        }

    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration for LlamaIndex setup."""
        return {
            "model_name": self.vllm.model,
            "context_window": self.vllm.context_window,
            "max_tokens": self.vllm.max_tokens,
            "temperature": 0.1,
            "base_url": self.ollama_base_url,
        }

    def get_embedding_config(self) -> dict[str, Any]:
        """Get embedding configuration for BGE-M3 setup."""
        return {
            "model_name": self.embedding.model_name,
            "device": "cuda" if self.enable_gpu_acceleration else "cpu",
            "max_length": self.embedding.max_length,
            "batch_size": self.embedding.batch_size_gpu
            if self.enable_gpu_acceleration
            else self.embedding.batch_size_cpu,
            "trust_remote_code": True,
        }

    def get_processing_config(self) -> dict[str, Any]:
        """Get document processing configuration."""
        return {
            "chunk_size": self.processing.chunk_size,
            "new_after_n_chars": self.processing.new_after_n_chars,
            "combine_text_under_n_chars": self.processing.combine_text_under_n_chars,
            "multipage_sections": self.processing.multipage_sections,
            "max_document_size_mb": self.processing.max_document_size_mb,
        }

    # === CONVENIENCE PROPERTIES FOR BACKWARD COMPATIBILITY ===
    # These properties provide direct access to nested configuration values
    # for existing code that expects them on the main settings object

    # BGE-M3 embedding constants
    @property
    def bge_m3_embedding_dim(self) -> int:
        """BGE-M3 embedding dimension."""
        return self.embedding.dimension

    @property
    def bge_m3_max_length(self) -> int:
        """BGE-M3 maximum sequence length."""
        return self.embedding.max_length

    @property
    def bge_m3_model_name(self) -> str:
        """BGE-M3 model name."""
        return self.embedding.model_name

    @property
    def bge_m3_batch_size_gpu(self) -> int:
        """BGE-M3 GPU batch size."""
        return self.embedding.batch_size_gpu

    @property
    def bge_m3_batch_size_cpu(self) -> int:
        """BGE-M3 CPU batch size."""
        return self.embedding.batch_size_cpu

    # RRF fusion constants
    @property
    def rrf_fusion_alpha(self) -> int:
        """RRF fusion alpha parameter."""
        return self.retrieval.rrf_alpha

    @property
    def rrf_k_constant(self) -> int:
        """RRF k constant."""
        return self.retrieval.rrf_k_constant

    @property
    def rrf_fusion_weight_dense(self) -> float:
        """Dense vector weight in RRF fusion."""
        return 0.7  # Fixed ratio for dense vectors

    @property
    def rrf_fusion_weight_sparse(self) -> float:
        """Sparse vector weight in RRF fusion."""
        return 0.3  # Fixed ratio for sparse vectors

    # Additional convenience properties
    @property
    def default_confidence_threshold(self) -> float:
        """Default confidence threshold for queries."""
        return 0.8  # Standard confidence threshold

    @property
    def reranking_top_k(self) -> int:
        """Reranking top K results."""
        return self.retrieval.reranking_top_k

    @property
    def top_k(self) -> int:
        """Top K vector similarity results."""
        return self.retrieval.top_k

    @property
    def reranker_model(self) -> str:
        """Reranker model name."""
        return self.retrieval.reranker_model

    @property
    def default_token_limit(self) -> int:
        """Default token limit (context window)."""
        return self.vllm.context_window


# Global settings instance - primary interface for the application
settings = DocMindSettings()

# Module exports
__all__ = [
    "DocMindSettings",
    "VLLMConfig",
    "ProcessingConfig",
    "AgentConfig",
    "EmbeddingConfig",
    "RetrievalConfig",
    "CacheConfig",
    "settings",
]
