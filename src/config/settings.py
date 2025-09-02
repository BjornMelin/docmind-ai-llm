"""Unified DocMind AI configuration using Pydantic Settings V2.

This module provides the main configuration architecture:
- Unified Pydantic BaseSettings with environment variable mapping
- Nested configuration models for complex areas

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
    temperature: float = Field(default=0.1, ge=0, le=2)

    # FP8 Optimization Settings
    gpu_memory_utilization: float = Field(default=0.85, ge=0.5, le=0.95)
    kv_cache_dtype: str = Field(default="fp8_e5m2")
    attention_backend: str = Field(default="FLASHINFER")
    enable_chunked_prefill: bool = Field(default=True)

    # Performance Settings
    max_num_seqs: int = Field(default=16, ge=1, le=64)
    max_num_batched_tokens: int = Field(default=8192, ge=1024, le=16384)

    # Backend Configuration
    backend: str = Field(
        default="ollama", description="LLM backend: vllm, ollama, llamacpp"
    )
    vllm_base_url: str = Field(
        default="http://localhost:8000", description="vLLM server endpoint"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama API endpoint"
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1", description="LM Studio API endpoint"
    )
    llamacpp_model_path: Path = Field(
        default=Path("./models/qwen3.gguf"),
        description="Path to GGUF model for llama.cpp backend",
    )


class ProcessingConfig(BaseModel):
    """Document processing configuration (ADR-009)."""

    # Unstructured.io Settings
    chunk_size: int = Field(default=1500, ge=100, le=10000)
    new_after_n_chars: int = Field(default=1200, ge=100, le=8000)
    combine_text_under_n_chars: int = Field(default=500, ge=50, le=2000)
    multipage_sections: bool = Field(default=True)
    chunk_overlap: int = Field(default=50, ge=0, le=200)
    max_document_size_mb: int = Field(default=100, ge=1, le=500)
    debug_chunk_flow: bool = Field(default=False)


class AgentConfig(BaseModel):
    """Multi-agent system configuration (ADR-011)."""

    enable_multi_agent: bool = Field(default=True)
    decision_timeout: int = Field(default=200, ge=10, le=1000)
    max_retries: int = Field(default=2, ge=0, le=10)
    max_concurrent_agents: int = Field(default=3, ge=1, le=10)
    enable_fallback_rag: bool = Field(default=True)

    # === ADR-011 AGENT CONTEXT MANAGEMENT ===
    context_trim_threshold: int = Field(default=122880, ge=65536, le=131072)
    context_buffer_size: int = Field(default=8192, ge=2048, le=131072)
    enable_parallel_tool_execution: bool = Field(default=True)
    max_workflow_depth: int = Field(default=5, ge=2, le=10)
    enable_agent_state_compression: bool = Field(default=True)
    chat_memory_limit_tokens: int = Field(default=66560, ge=32768, le=98304)


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
    reranker_normalize_scores: bool = Field(default=True)

    # RRF Fusion Settings
    rrf_alpha: int = Field(default=60, ge=10, le=100)
    rrf_k_constant: int = Field(default=60, ge=10, le=100)
    rrf_fusion_weight_dense: float = Field(default=0.7, ge=0, le=1)
    rrf_fusion_weight_sparse: float = Field(default=0.3, ge=0, le=1)
    use_sparse_embeddings: bool = Field(default=True)


class CacheConfig(BaseModel):
    """Simple cache configuration."""

    enable_document_caching: bool = Field(default=True)
    ttl_seconds: int = Field(default=3600, ge=300, le=86400)
    max_size_mb: int = Field(default=1000, ge=100, le=10000)
    enable_semantic_cache: bool = Field(default=True)
    semantic_threshold: float = Field(default=0.85, ge=0.5, le=0.95)


class DatabaseConfig(BaseModel):
    """Database and vector store configuration."""

    # Vector Database
    vector_store_type: str = Field(default="qdrant")
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="docmind_docs")
    qdrant_timeout: int = Field(default=60, ge=10, le=300)

    # SQL Database
    sqlite_db_path: Path = Field(default=Path("./data/docmind.db"))
    enable_wal_mode: bool = Field(default=True)


class UIConfig(BaseModel):
    """User interface configuration."""

    # Streamlit Configuration
    streamlit_port: int = Field(default=8501, ge=1024, le=65535)
    enable_dark_mode: bool = Field(default=True)
    theme: str = Field(default="auto")
    enable_session_persistence: bool = Field(default=True)
    response_streaming: bool = Field(default=True)
    max_history_items: int = Field(default=100, ge=10, le=1000)

    # User Interface Options
    context_size_options: list[int] = Field(
        default=[4096, 8192, 16384, 32768, 65536, 131072]
    )
    request_timeout_seconds: int = Field(default=30, ge=5, le=300)
    streaming_delay_seconds: float = Field(default=0.1, ge=0.01, le=2)
    default_token_limit: int = Field(default=8192, ge=1024, le=131072)


class MonitoringConfig(BaseModel):
    """Performance monitoring and system metrics configuration."""

    # Performance Limits
    max_query_latency_ms: int = Field(default=2000, ge=100, le=10000)
    max_memory_gb: float = Field(default=4.0, ge=1, le=64)
    max_vram_gb: float = Field(default=14.0, ge=2, le=48)
    enable_performance_logging: bool = Field(default=True)

    # System Metrics
    cpu_monitoring_interval: float = Field(default=0.1, ge=0.01, le=1)
    default_batch_size: int = Field(default=20, ge=1, le=100)
    default_confidence_threshold: float = Field(default=0.8, ge=0, le=1)

    # Memory Conversion Constants
    bytes_to_gb_divisor: int = Field(default=1024**3)
    bytes_to_mb_divisor: int = Field(default=1024**2)
    percent_multiplier: int = Field(default=100)

    # Cache and Timeout Settings
    cache_expiry_seconds: int = Field(default=3600, ge=300, le=86400)
    spacy_download_timeout: int = Field(default=300, ge=60, le=600)
    default_agent_timeout: float = Field(default=3.0, ge=0.5, le=30)
    default_entity_confidence: float = Field(default=0.8, ge=0, le=1)


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
        extra="ignore",  # CRITICAL: Changed from "forbid" to fix validation errors
    )

    # Core Application
    app_name: str = Field(default="DocMind AI")
    app_version: str = Field(default="2.0.0")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # File System Paths
    data_dir: Path = Field(default=Path("./data"))
    cache_dir: Path = Field(default=Path("./cache"))
    log_file: Path = Field(default=Path("./logs/docmind.log"))

    # Analytics (ADR-032)
    analytics_enabled: bool = Field(
        default=False, description="Enable optional local DuckDB analytics database"
    )
    analytics_retention_days: int = Field(
        default=60, ge=1, le=365, description="Days to retain analytics records"
    )
    analytics_db_path: Path | None = Field(
        default=None,
        description=(
            "Optional override path; default is data_dir/analytics/analytics.duckdb"
        ),
    )

    # Backup (ADR-033)
    backup_enabled: bool = Field(
        default=False, description="Enable manual local backups with simple rotation"
    )
    backup_keep_last: int = Field(
        default=7,
        ge=1,
        le=100,
        description="How many backups to retain during rotation",
    )

    # Backend Configuration
    llm_backend: str = Field(default="ollama")
    ollama_base_url: str = Field(default="http://localhost:11434")
    enable_gpu_acceleration: bool = Field(default=True)

    # Advanced Features
    enable_graphrag: bool = Field(default=False)
    enable_dspy_optimization: bool = Field(
        default=False
    )  # Changed to False per user feedback
    enable_multimodal: bool = Field(default=False)
    analysis_modes: str = Field(default="detailed,summary,comparison")

    # === DSPY OPTIMIZATION PARAMETERS ===
    dspy_optimization_iterations: int = Field(default=10, ge=5, le=50)
    dspy_optimization_samples: int = Field(default=20, ge=5, le=100)
    dspy_max_retries: int = Field(default=3, ge=1, le=10)
    dspy_temperature: float = Field(default=0.1, ge=0, le=2)
    dspy_metric_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    enable_dspy_bootstrapping: bool = Field(default=True)

    # === GRAPHRAG CONFIGURATION ===
    graphrag_relationship_extraction: bool = Field(default=False)
    graphrag_entity_resolution: str = Field(default="fuzzy")
    graphrag_max_hops: int = Field(default=2, ge=1, le=5)
    graphrag_max_triplets: int = Field(default=1000, ge=100, le=10000)
    graphrag_chunk_size: int = Field(default=1024, ge=256, le=4096)

    # UI Configuration moved to nested UIConfig structure

    # Nested Configuration Models
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    def model_post_init(self, __context: Any) -> None:
        """Create necessary directories after initialization."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.database.sqlite_db_path.parent != self.data_dir:
            self.database.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

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
            "temperature": self.vllm.temperature,
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

    # === ADR COMPLIANCE CONFIGURATION METHODS ===

    def get_agent_orchestration_config(self) -> dict[str, Any]:
        """Get agent orchestration configuration for ADR-011 compliance."""
        return {
            "context_trim_threshold": self.agents.context_trim_threshold,
            "context_buffer_size": self.agents.context_buffer_size,
            "enable_parallel_execution": self.agents.enable_parallel_tool_execution,
            "max_workflow_depth": self.agents.max_workflow_depth,
            "enable_state_compression": self.agents.enable_agent_state_compression,
            "chat_memory_limit": self.agents.chat_memory_limit_tokens,
            "decision_timeout": self.agents.decision_timeout,
        }

    def get_dspy_config(self) -> dict[str, Any]:
        """Get DSPy optimization configuration for ADR-018."""
        return {
            "enabled": self.enable_dspy_optimization,
            "iterations": self.dspy_optimization_iterations,
            "samples": self.dspy_optimization_samples,
            "max_retries": self.dspy_max_retries,
            "temperature": self.dspy_temperature,
            "metric_threshold": self.dspy_metric_threshold,
            "bootstrapping": self.enable_dspy_bootstrapping,
        }

    def get_graphrag_config(self) -> dict[str, Any]:
        """Get GraphRAG configuration for ADR-019."""
        return {
            "enabled": self.enable_graphrag,
            "relationship_extraction": self.graphrag_relationship_extraction,
            "entity_resolution": self.graphrag_entity_resolution,
            "max_hops": self.graphrag_max_hops,
            "max_triplets": self.graphrag_max_triplets,
            "chunk_size": self.graphrag_chunk_size,
        }

    def get_ui_config(self) -> dict[str, Any]:
        """Get UI configuration for ADR-016."""
        return {
            "theme": self.ui.theme,
            "session_persistence": self.ui.enable_session_persistence,
            "response_streaming": self.ui.response_streaming,
            "max_history_items": self.ui.max_history_items,
            "streamlit_port": self.ui.streamlit_port,
            "enable_dark_mode": self.ui.enable_dark_mode,
        }


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
    "DatabaseConfig",
    "UIConfig",
    "MonitoringConfig",
    "settings",
]


def get_vllm_env_vars() -> dict[str, str]:
    """Module-level proxy for tests to patch vLLM env vars provider.

    Returns:
        dict[str, str]: Environment variables for vLLM processes.
    """
    return settings.get_vllm_env_vars()
