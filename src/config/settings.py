"""Unified DocMind AI configuration using Pydantic Settings v2.

Provides a typed, nested configuration model with environment variable
mapping. Prefer nested fields and `DOCMIND_{SECTION}__{FIELD}` env vars.

Usage:
    from src.config.settings import settings
    print(settings.embedding.model_name)
"""

from contextlib import suppress
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

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

    vllm_base_url: str = Field(
        default="http://localhost:8000", description="vLLM server endpoint"
    )
    llamacpp_model_path: Path = Field(
        default=Path("./models/qwen3.gguf"),
        description="Path to GGUF model for llama.cpp backend",
    )


class SemanticCacheConfig(BaseModel):
    """Application-level semantic cache configuration (ADR-035)."""

    enabled: bool = Field(default=False)
    provider: Literal["gptcache", "none"] = Field(default="gptcache")
    score_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    ttl_seconds: int = Field(default=1_209_600, ge=60)  # 14 days
    top_k: int = Field(default=5, ge=1, le=50)
    max_response_bytes: int = Field(default=24_000, ge=1024)
    namespace: str = Field(default="default")


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


class ChatConfig(BaseModel):
    """Chat memory configuration (ADR-021)."""

    sqlite_path: Path = Field(default=Path("./data/docmind.db"))


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


class AnalysisConfig(BaseModel):
    """Analysis mode settings (ADR-023)."""

    mode: Literal["auto", "separate", "combined"] = Field(default="auto")
    max_workers: int = Field(default=4, ge=1, le=32)


class EmbeddingConfig(BaseModel):
    """Embedding configuration (SPEC-003; ADR-002/004).

    Text uses BGE-M3. Images use tiered backbones (OpenCLIP/SigLIP) with
    hardware-aware defaults. All knobs remain optional and conservative.
    """

    # Text (BGE-M3)
    model_name: str = Field(default="BAAI/bge-m3")
    dimension: int = Field(default=1024, ge=256, le=4096)
    max_length: int = Field(default=8192, ge=512, le=16384)
    enable_sparse: bool = Field(default=True)
    normalize_text: bool = Field(default=True)
    batch_size_text_gpu: int = Field(default=12, ge=1, le=128)
    batch_size_text_cpu: int = Field(default=4, ge=1, le=64)

    # Images
    image_backbone: Literal[
        "auto",
        "openclip_vitl14",
        "openclip_vith14",
        "siglip_base",
        "bge_visualized",
    ] = Field(default="auto")
    normalize_image: bool = Field(default=True)
    batch_size_image: int = Field(default=8, ge=1, le=64)

    # Device selection (auto|cpu|cuda)
    embed_device: Literal["auto", "cpu", "cuda"] = Field(default="auto")


class RetrievalConfig(BaseModel):
    """Retrieval and reranking configuration (ADR-006)."""

    strategy: str = Field(default="hybrid")
    top_k: int = Field(default=10, ge=1, le=50)
    use_reranking: bool = Field(default=True)
    reranking_top_k: int = Field(default=5, ge=1, le=20)
    reranker_normalize_scores: bool = Field(default=True)
    # ADR-024/036/037: explicit reranker mode selection
    # auto: modality-aware (visual+text); text: text-only; multimodal: force both
    reranker_mode: Literal["auto", "text", "multimodal"] = Field(default="auto")

    # RRF Fusion Settings
    rrf_alpha: int = Field(default=60, ge=10, le=100)
    rrf_k_constant: int = Field(default=60, ge=10, le=100)
    rrf_fusion_weight_dense: float = Field(default=0.7, ge=0, le=1)
    rrf_fusion_weight_sparse: float = Field(default=0.3, ge=0, le=1)
    use_sparse_embeddings: bool = Field(default=True)
    # Feature flag for future named-vectors multi-head support (no-op now)
    named_vectors_multi_head_enabled: bool = Field(default=False)
    # Router and feature toggles (ADR-003)
    router: Literal["auto", "simple", "hierarchical", "graph"] = Field(default="auto")
    hybrid_enabled: bool = Field(default=True)
    hierarchy_enabled: bool = Field(default=True)
    graph_enabled: bool = Field(default=False)
    # Reranker model
    # (text-only CrossEncoder; ADR-006 legacy, ADR-037 multimodal supersedes)
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    # Optional keyword tool (BM25) registration flag (disabled by default)
    enable_keyword_tool: bool = Field(default=False)


class CacheConfig(BaseModel):
    """Document processing cache toggles (ADR-030)."""

    enable_document_caching: bool = Field(default=True)
    ttl_seconds: int = Field(default=3600, ge=300, le=86400)
    max_size_mb: int = Field(default=1000, ge=100, le=10000)
    # Path configuration for DuckDB KV store
    dir: Path = Field(default=Path("./cache"))
    filename: str = Field(default="docmind.duckdb")


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


class GraphRAGConfig(BaseModel):
    """GraphRAG configuration (ADR-019)."""

    enabled: bool = Field(default=True)
    relationship_extraction: bool = Field(default=False)
    entity_resolution: Literal["fuzzy", "exact"] = Field(default="fuzzy")
    max_hops: int = Field(default=2, ge=1, le=5)
    max_triplets: int = Field(default=1000, ge=100, le=10000)
    chunk_size: int = Field(default=1024, ge=128, le=8192)


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
        extra="ignore",
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

    # Backend Configuration (strict, supported only)
    llm_backend: Literal["vllm", "ollama", "lmstudio", "llamacpp"] = Field(
        default="ollama"
    )
    # Top-level model controls (mirrors nested vllm.* when provided)
    model: str | None = Field(
        default=None,
        description=(
            "Preferred model identifier (or GGUF path for LlamaCPP). If set,"
            " overrides nested vllm.model at runtime."
        ),
    )
    context_window: int | None = Field(
        default=None,
        ge=1024,
        le=200000,
        description=(
            "Global context window cap. If set, overrides nested vllm.context_window"
        ),
    )
    ollama_base_url: str = Field(default="http://localhost:11434")
    lmstudio_base_url: str = Field(default="http://localhost:1234/v1")
    vllm_base_url: str | None = Field(
        default=None, description="vLLM server endpoint (OpenAI or native)"
    )
    llamacpp_base_url: str | None = Field(
        default=None, description="Optional llama.cpp server (OpenAI-compatible)"
    )
    enable_gpu_acceleration: bool = Field(default=True)

    # Security for endpoints
    allow_remote_endpoints: bool = Field(
        default=False,
        description=(
            "When False, only localhost/127.0.0.1 endpoints are allowed for LLMs"
        ),
    )
    endpoint_allowlist: list[str] = Field(
        default_factory=lambda: [
            "http://localhost",
            "http://127.0.0.1",
            "https://localhost",
            "https://127.0.0.1",
        ]
    )

    # Feature flags
    guided_json_enabled: bool = Field(
        default=False, description="Structured outputs available (SPEC-007 prep)"
    )

    # OpenAI-compatible client flags (for LM Studio, vLLM OpenAI mode, llama-cpp server)
    openai_like_api_key: str = Field(default="not-needed")
    openai_like_is_chat_model: bool = Field(default=True)
    openai_like_is_function_calling_model: bool = Field(default=False)
    openai_like_extra_headers: dict | None = Field(default=None)

    # Global LLM client behavior
    llm_request_timeout_seconds: int = Field(default=120, ge=5, le=600)
    llm_streaming_enabled: bool = Field(default=True)

    # Advanced Features
    enable_graphrag: bool = Field(
        default=True,
        description=(
            "GraphRAG configuration (ADR-019). Breaking change: default is True "
            "(was False). Disable explicitly to revert to prior behavior."
        ),
    )
    enable_dspy_optimization: bool = Field(
        default=False
    )  # Changed to False per user feedback
    enable_multimodal: bool = Field(default=False)

    # DSPy optimization parameters
    dspy_optimization_iterations: int = Field(default=10, ge=5, le=50)
    dspy_optimization_samples: int = Field(default=20, ge=5, le=100)
    dspy_max_retries: int = Field(default=3, ge=1, le=10)
    dspy_temperature: float = Field(default=0.1, ge=0, le=2)
    dspy_metric_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    enable_dspy_bootstrapping: bool = Field(default=True)

    # GraphRAG legacy (prefer nested graphrag_cfg)
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
    # Additional nested configs (ADR-024)
    semantic_cache: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    graphrag_cfg: GraphRAGConfig = Field(default_factory=GraphRAGConfig)

    # Compatibility alias fields mapped to nested configs. These allow
    # ergonomic top-level overrides via environment variables.
    context_window_size: int | None = Field(default=None)
    chunk_size: int | None = Field(default=None)
    chunk_overlap: int | None = Field(default=None)
    enable_multi_agent: bool | None = Field(default=None)
    # Top-level LLM context window cap (ADR-004/024)
    llm_context_window_max: int = Field(default=131072, ge=8192, le=200000)

    def model_post_init(self, __context: Any) -> None:
        """Create necessary directories after initialization."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.database.sqlite_db_path.parent != self.data_dir:
            self.database.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        # Apply compatibility alias fields to nested configs if provided
        # Keep nested vllm.* in sync with top-level overrides when present
        if self.model:
            with suppress(Exception):
                self.vllm.model = str(self.model)
        if self.context_window:
            if int(self.context_window) <= 0:
                raise ValueError("context_window must be > 0")
            with suppress(Exception):
                self.vllm.context_window = int(self.context_window)
        if self.vllm_base_url:
            with suppress(Exception):
                self.vllm.vllm_base_url = str(self.vllm_base_url)
        if self.context_window_size is not None:
            if int(self.context_window_size) <= 0:
                raise ValueError("context_window_size must be > 0")
            with suppress(Exception):
                self.vllm.context_window = int(self.context_window_size)
        if self.chunk_size is not None:
            if int(self.chunk_size) <= 0:
                raise ValueError("chunk_size must be > 0")
            with suppress(Exception):
                self.processing.chunk_size = int(self.chunk_size)
        if self.chunk_overlap is not None:
            if self.chunk_size is not None and int(self.chunk_overlap) > int(
                self.chunk_size
            ):
                raise ValueError("chunk_overlap cannot exceed chunk_size")
            with suppress(Exception):
                self.processing.chunk_overlap = int(self.chunk_overlap)
        if self.enable_multi_agent is not None:
            with suppress(Exception):
                self.agents.enable_multi_agent = bool(self.enable_multi_agent)

        # Validate base URLs and security posture
        self._validate_endpoints_security()
        self._validate_lmstudio_url()

    def get_vllm_config(self) -> dict[str, Any]:  # pragma: no cover - simple proxy
        """Get vLLM configuration for client setup."""
        return {
            "model": self.model or self.vllm.model,
            "context_window": int(self.context_window or self.vllm.context_window),
            "temperature": self.vllm.temperature,
        }

    def get_agent_config(self) -> dict[str, Any]:  # pragma: no cover - simple proxy
        """Return agent configuration as a simple mapping."""
        return {
            "decision_timeout": self.agents.decision_timeout,
            "max_retries": self.agents.max_retries,
            "enable_multi_agent": self.agents.enable_multi_agent,
        }

    def get_vllm_env_vars(self) -> dict[str, str]:
        """Return environment variables for vLLM process setup."""
        return {
            "VLLM_ATTENTION_BACKEND": self.vllm.attention_backend,
            "VLLM_KV_CACHE_DTYPE": self.vllm.kv_cache_dtype,
            "VLLM_GPU_MEMORY_UTILIZATION": str(self.vllm.gpu_memory_utilization),
            "VLLM_ENABLE_CHUNKED_PREFILL": "1"
            if self.vllm.enable_chunked_prefill
            else "0",
            # Provide max model len for launcher scripts that read env
            "VLLM_MAX_MODEL_LEN": str(self.llm_context_window_max),
        }

    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration for LlamaIndex setup."""
        return {
            "model_name": self.model or self.vllm.model,
            # Enforce cap consistently
            "context_window": min(
                int(self.context_window or self.vllm.context_window),
                self.llm_context_window_max,
            ),
            "max_tokens": self.vllm.max_tokens,
            "temperature": self.vllm.temperature,
            "base_url": self.ollama_base_url,
        }

    def get_embedding_config(self) -> dict[str, Any]:
        """Get embedding configuration for embedding factories.

        Returns a flat mapping used by factory helpers while keeping the
        class-based configuration as the single source of truth.
        """
        device = (
            ("cuda" if self.enable_gpu_acceleration else "cpu")
            if self.embedding.embed_device == "auto"
            else self.embedding.embed_device
        )
        return {
            # Text
            "model_name": self.embedding.model_name,
            "device": device,
            "dimension": self.embedding.dimension,
            "max_length": self.embedding.max_length,
            "batch_size_text": (
                self.embedding.batch_size_text_gpu
                if device == "cuda"
                else self.embedding.batch_size_text_cpu
            ),
            # Expose raw batch sizes for downstream consumers
            "batch_size_text_gpu": self.embedding.batch_size_text_gpu,
            "batch_size_text_cpu": self.embedding.batch_size_text_cpu,
            "normalize_text": self.embedding.normalize_text,
            "enable_sparse": self.embedding.enable_sparse,
            # Images
            "image_backbone": self.embedding.image_backbone,
            "batch_size_image": self.embedding.batch_size_image,
            "normalize_image": self.embedding.normalize_image,
            # Misc
            "embed_device": self.embedding.embed_device,
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

    def get_cache_db_path(self) -> Path:
        """Return full path to DuckDB KV store file (ADR-030)."""
        return (self.cache.dir or self.cache_dir) / self.cache.filename

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
        # Prefer nested configuration; fallback to legacy top-level fields
        if hasattr(self, "graphrag_cfg") and isinstance(
            self.graphrag_cfg, GraphRAGConfig
        ):
            c = self.graphrag_cfg
            return {
                "enabled": c.enabled,
                "relationship_extraction": c.relationship_extraction,
                "entity_resolution": c.entity_resolution,
                "max_hops": c.max_hops,
                "max_triplets": c.max_triplets,
                "chunk_size": c.chunk_size,
            }
        return {
            "enabled": self.enable_graphrag,
            "relationship_extraction": self.graphrag_relationship_extraction,
            "entity_resolution": self.graphrag_entity_resolution,
            "max_hops": self.graphrag_max_hops,
            "max_triplets": self.graphrag_max_triplets,
            "chunk_size": self.graphrag_chunk_size,
        }

    def get_semantic_cache_config(self) -> dict[str, Any]:
        """Semantic cache configuration for ADR-035."""
        c = self.semantic_cache
        return {
            "enabled": c.enabled,
            "provider": c.provider,
            "score_threshold": c.score_threshold,
            "ttl_seconds": c.ttl_seconds,
            "top_k": c.top_k,
            "max_response_bytes": c.max_response_bytes,
            "namespace": c.namespace,
        }

    def get_chat_config(self) -> dict[str, Any]:
        """Chat memory configuration for ADR-021."""
        return {"sqlite_path": str(self.chat.sqlite_path)}

    def get_analysis_config(self) -> dict[str, Any]:
        """Analysis mode configuration for ADR-023."""
        return {
            "mode": self.analysis.mode,
            "max_workers": self.analysis.max_workers,
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

    # === Validation helpers ===
    def _validate_endpoints_security(self) -> None:
        """Validate endpoint URLs against security policy.

        When ``allow_remote_endpoints`` is False, only localhost/127.0.0.1
        URLs are permitted. Users may extend ``endpoint_allowlist``.
        """
        if self.allow_remote_endpoints:
            return

        def _is_allowed(url: str | None) -> bool:
            if not url:
                return True
            try:
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    return True  # empty or malformed handled elsewhere
                for prefix in self.endpoint_allowlist:
                    if url.startswith(prefix):
                        return True
                # Also accept explicit localhost with ports
                host = parsed.hostname or ""
                return host in {"localhost", "127.0.0.1"}
            except (ValueError, TypeError):  # pragma: no cover - defensive
                return False

        urls = [
            self.ollama_base_url,
            self.lmstudio_base_url,
            self.vllm_base_url or self.vllm.vllm_base_url,
            self.llamacpp_base_url,
        ]
        for url in urls:
            if not _is_allowed(url):
                raise ValueError(
                    "Remote endpoints are disabled. Set allow_remote_endpoints=True "
                    "or use localhost URLs."
                )

    def _validate_lmstudio_url(self) -> None:
        """Ensure LM Studio base URL ends with /v1 as required by its API."""
        if self.lmstudio_base_url and not self.lmstudio_base_url.rstrip("/").endswith(
            "v1"
        ):
            raise ValueError("LM Studio base URL must end with /v1")


# Global settings instance - primary interface for the application
settings = DocMindSettings()

# Module exports
__all__ = [
    "AgentConfig",
    "AnalysisConfig",
    "CacheConfig",
    "ChatConfig",
    "DatabaseConfig",
    "DocMindSettings",
    "EmbeddingConfig",
    "GraphRAGConfig",
    "MonitoringConfig",
    "ProcessingConfig",
    "RetrievalConfig",
    "SemanticCacheConfig",
    "UIConfig",
    "VLLMConfig",
    "settings",
]


def get_vllm_env_vars() -> dict[str, str]:
    """Module-level helper returning vLLM environment variables."""
    return settings.get_vllm_env_vars()
