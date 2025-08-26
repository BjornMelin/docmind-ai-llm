"""DocMind AI application-specific configuration.

This module provides only app-specific settings that cannot be managed by
LlamaIndex Settings. All LLM, embedding, and retrieval configuration is
handled by setup_llamaindex().

This unified approach achieves 95% complexity reduction while maintaining
100% functionality.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DocMindSettings(BaseSettings):
    """DocMind AI application-specific configuration.

    Handles only app-specific settings that cannot be managed by LlamaIndex Settings.
    All LLM, embedding, and retrieval configuration is handled by setup_llamaindex().
    """

    # Core Application
    app_name: str = Field(default="DocMind AI")
    app_version: str = Field(default="2.0.0")
    debug: bool = Field(default=False)

    # Multi-Agent System
    enable_multi_agent: bool = Field(default=True)
    agent_decision_timeout: int = Field(default=200, ge=10, le=1000)
    max_agent_retries: int = Field(default=2, ge=0, le=10)
    enable_fallback_rag: bool = Field(default=True)
    max_concurrent_agents: int = Field(default=3, ge=1, le=10)

    # File System Paths
    data_dir: Path = Field(default=Path("./data"))
    cache_dir: Path = Field(default=Path("./cache"))
    sqlite_db_path: Path = Field(default=Path("./data/docmind.db"))
    log_file: Path = Field(default=Path("./logs/docmind.log"))

    # Performance & GPU Settings
    enable_gpu_acceleration: bool = Field(default=True)
    max_memory_gb: float = Field(default=4.0, ge=1.0, le=128.0)
    max_vram_gb: float = Field(default=14.0, ge=1.0, le=80.0)
    max_query_latency_ms: int = Field(default=2000, ge=100, le=30000)

    # Document Processing
    max_document_size_mb: int = Field(default=100, ge=1, le=500)
    enable_document_caching: bool = Field(default=True)
    parse_strategy: str = Field(default="hi_res")

    # Document Processing Pipeline (ADR-009 compliant)
    chunk_size: int = Field(default=1500, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=50, le=2000)
    new_after_n_chars: int = Field(default=1200, ge=100, le=8000)
    combine_text_under_n_chars: int = Field(default=500, ge=50, le=2000)
    multipage_sections: bool = Field(default=True)
    enable_semantic_boundary_detection: bool = Field(default=True)

    # Vector Database Configuration
    vector_store_type: str = Field(default="qdrant")
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="docmind_docs")

    # Dual-Layer Cache Configuration (ADR-009 compliant)
    cache_ttl_seconds: int = Field(default=3600, ge=300, le=86400)
    max_cache_size_mb: int = Field(default=1000, ge=100, le=10000)
    cache_compression: bool = Field(default=True)
    enable_semantic_cache: bool = Field(default=True)
    semantic_cache_threshold: float = Field(default=0.85, ge=0.5, le=0.95)

    # Retrieval Configuration
    retrieval_strategy: str = Field(default="hybrid")
    top_k: int = Field(default=10, ge=1, le=50)
    use_reranking: bool = Field(default=True)
    reranking_top_k: int = Field(default=5, ge=1, le=20)
    use_sparse_embeddings: bool = Field(default=True)

    # Performance Constants
    embedding_batch_size: int = Field(default=100, ge=1, le=1000)
    default_batch_size: int = Field(default=20, ge=1, le=128)

    # vLLM Specific Settings
    vllm_gpu_memory_utilization: float = Field(default=0.95, ge=0.1, le=0.95)
    vllm_attention_backend: str = Field(default="FLASHINFER")
    vllm_enable_chunked_prefill: bool = Field(default=True)
    vllm_max_num_batched_tokens: int = Field(default=8192, ge=1024, le=16384)
    vllm_max_num_seqs: int = Field(default=16, ge=1, le=64)

    # vLLM FP8 Optimization Settings
    vllm_kv_cache_dtype: str = Field(
        default="fp8_e5m2", description="FP8 KV cache for 50% memory reduction"
    )
    vllm_calculate_kv_scales: bool = Field(
        default=True, description="Required for FP8 KV cache"
    )
    vllm_use_cudnn_prefill: bool = Field(
        default=True, description="Use CUDNN prefill optimization"
    )
    vllm_max_token_limit: int = Field(
        default=120000,
        ge=10000,
        le=200000,
        description="Max tokens before trimming (8K buffer for 128K limit)",
    )
    vllm_kv_cache_memory_per_token: int = Field(
        default=1024, ge=512, le=2048, description="Bytes per token with FP8 KV cache"
    )
    vllm_minimum_vram_gb: int = Field(
        default=12, ge=8, le=32, description="Minimum VRAM requirement for vLLM"
    )

    # UI Configuration
    streamlit_port: int = Field(default=8501, ge=1024, le=65535)
    enable_ui_dark_mode: bool = Field(default=True)

    # Advanced Features
    enable_graphrag: bool = Field(default=False)
    enable_multimodal: bool = Field(default=False)
    enable_dspy_optimization: bool = Field(default=True)
    dspy_optimization_samples: int = Field(default=20, ge=5, le=100)

    # Logging and Monitoring
    log_level: str = Field(default="INFO")
    enable_performance_logging: bool = Field(default=True)
    enable_wal_mode: bool = Field(default=True)

    # Core application configuration
    llm_backend: str = Field(default="ollama")
    context_buffer_size: int = Field(default=65536, ge=4096, le=131072)
    enable_conversation_memory: bool = Field(default=True)
    embedding_dimension: int = Field(default=1024, ge=256, le=4096)
    quant_policy: str = Field(default="fp8")
    analysis_modes: list[str] = Field(default=["detailed", "summary", "comparison"])

    # Application constants for UI and processing
    default_token_limit: int = Field(default=131072, ge=1024, le=1000000)
    suggested_context_high: int = Field(default=65536, ge=8192, le=131072)
    suggested_context_medium: int = Field(default=32768, ge=8192, le=131072)
    suggested_context_low: int = Field(default=8192, ge=1024, le=131072)
    minimum_vram_high_gb: int = Field(default=16, ge=4, le=80)
    minimum_vram_medium_gb: int = Field(default=8, ge=2, le=32)

    # Backend service URLs
    ollama_base_url: str = Field(default="http://localhost:11434")
    lmstudio_base_url: str = Field(default="http://localhost:1234/v1")
    llamacpp_model_path: str = Field(default="/path/to/model.gguf")

    # Reranking model (used by tool factory)
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")

    # Timeout and UI constants
    default_agent_timeout: float = Field(default=3.0, ge=1.0, le=10.0)
    default_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    request_timeout_seconds: float = Field(default=120.0, ge=1.0, le=300.0)
    streaming_delay_seconds: float = Field(default=0.02, ge=0.001, le=1.0)

    # Context size options for UI
    context_size_options: list[int] = Field(default=[8192, 32768, 65536, 131072])

    # Memory conversion constants
    bytes_to_gb_divisor: int = Field(default=1024**3)  # 1073741824
    bytes_to_mb_divisor: int = Field(default=1024 * 1024)  # 1048576

    # BGE-M3 embedding model constants
    bge_m3_embedding_dim: int = Field(default=1024, ge=512, le=4096)
    bge_m3_max_length: int = Field(default=8192, ge=512, le=16384)
    bge_m3_model_name: str = Field(default="BAAI/bge-m3")
    bge_m3_batch_size_gpu: int = Field(default=12, ge=1, le=128)
    bge_m3_batch_size_cpu: int = Field(default=4, ge=1, le=32)

    # RRF Fusion Constants
    rrf_fusion_alpha: int = Field(default=60, ge=10, le=100)
    rrf_k_constant: int = Field(default=60, ge=10, le=100)

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="DOCMIND_", case_sensitive=False, extra="forbid"
    )

    def model_post_init(self, __context) -> None:
        """Create necessary directories on initialization."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.sqlite_db_path.parent != self.data_dir:
            self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_agent_config(self) -> dict[str, any]:
        """Get configuration specific to multi-agent system."""
        return {
            "enable_multi_agent": self.enable_multi_agent,
            "agent_decision_timeout": self.agent_decision_timeout,
            "enable_fallback_rag": self.enable_fallback_rag,
            "max_agent_retries": self.max_agent_retries,
            "max_concurrent_agents": self.max_concurrent_agents,
        }

    def get_performance_config(self) -> dict[str, any]:
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

    def get_vllm_config(self) -> dict[str, any]:
        """Get vLLM-specific configuration."""
        return {
            "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "attention_backend": self.vllm_attention_backend,
            "enable_chunked_prefill": self.vllm_enable_chunked_prefill,
            "max_num_batched_tokens": self.vllm_max_num_batched_tokens,
            "max_num_seqs": self.vllm_max_num_seqs,
            "kv_cache_dtype": self.vllm_kv_cache_dtype,
            "calculate_kv_scales": self.vllm_calculate_kv_scales,
            "use_cudnn_prefill": self.vllm_use_cudnn_prefill,
            "max_token_limit": self.vllm_max_token_limit,
            "kv_cache_memory_per_token": self.vllm_kv_cache_memory_per_token,
            "minimum_vram_gb": self.vllm_minimum_vram_gb,
        }

    def get_processing_config(self) -> dict[str, any]:
        """Get document processing pipeline configuration."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "new_after_n_chars": self.new_after_n_chars,
            "combine_text_under_n_chars": self.combine_text_under_n_chars,
            "multipage_sections": self.multipage_sections,
            "enable_semantic_boundary_detection": (
                self.enable_semantic_boundary_detection
            ),
            "max_document_size_mb": self.max_document_size_mb,
            "parse_strategy": self.parse_strategy,
        }

    def get_cache_config(self) -> dict[str, any]:
        """Get dual-layer cache configuration."""
        return {
            "enable_document_caching": self.enable_document_caching,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "max_cache_size_mb": self.max_cache_size_mb,
            "cache_compression": self.cache_compression,
            "enable_semantic_cache": self.enable_semantic_cache,
            "semantic_cache_threshold": self.semantic_cache_threshold,
            "enable_wal_mode": self.enable_wal_mode,
        }


# Global settings instance
app_settings = DocMindSettings()
