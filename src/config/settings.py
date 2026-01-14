"""Unified DocMind AI configuration using Pydantic Settings v2.

Provides a typed, nested configuration model with environment variable
mapping. Prefer nested fields and `DOCMIND_{SECTION}__{FIELD}` env vars.

Usage:
    from src.config.settings import settings
    print(settings.embedding.model_name)
"""

import logging
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.models.embedding_constants import ImageBackboneName

logger = logging.getLogger(__name__)

SETTINGS_MODEL_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_prefix="DOCMIND_",
    env_nested_delimiter="__",
    case_sensitive=False,
    extra="ignore",
)


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
    provider: Literal["qdrant", "none"] = Field(default="qdrant")
    collection_name: str | None = Field(default=None)
    score_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    ttl_seconds: int = Field(default=1_209_600, ge=0)  # 14 days
    top_k: int = Field(default=5, ge=1)
    max_response_bytes: int = Field(default=24_000, ge=0)
    namespace: str = Field(default="default")
    allow_semantic_for_templates: list[str] | None = Field(default=None)


def ensure_v1(url: str | None) -> str | None:
    """Normalize OpenAI-compatible base URLs to include a single /v1 segment."""
    if not url:
        return url
    try:
        parsed = urlparse(url.rstrip("/"))
        path = parsed.path or ""
        if not path.endswith("/v1"):
            path = f"{path}/v1"
        return parsed._replace(path=path).geturl()
    except (ValueError, AttributeError, TypeError):
        return url


class OpenAIConfig(BaseModel):
    """Settings for OpenAI-compatible servers.

    Used for LM Studio, vLLM OpenAI-compatible server, and llama.cpp server.
    Ensures idempotent normalization of base URLs to include a single "/v1".
    """

    base_url: str = Field(default="http://localhost:1234/v1")
    api_key: str | None = Field(default=None, description="Optional API key")

    @field_validator("base_url", mode="before")
    @classmethod
    def _ensure_v1_on_base(cls, v: str) -> str:
        return ensure_v1((v or "").strip()) or ""


_DEFAULT_OPENAI_BASE_URL = OpenAIConfig().base_url


class ImageConfig(BaseModel):
    """Image processing and security settings."""

    img_aes_key_base64: str | None = Field(default=None)
    img_kid: str | None = Field(default=None)
    img_delete_plaintext: bool = Field(default=False)


class SecurityConfig(BaseModel):
    """Security and remote endpoint policy settings."""

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
    trust_remote_code: bool = Field(
        default=False,
        description="Default posture for libraries that support remote code execution",
    )


class HybridConfig(BaseModel):
    """Declarative hybrid retrieval policy."""

    enabled: bool = Field(default=False)
    server_side: bool = Field(default=False)
    method: Literal["rrf", "dbsf"] = Field(default="rrf")
    rrf_k: int = Field(default=60, ge=1, le=256)
    dbsf_alpha: float = Field(default=0.5, ge=0.0, le=1.0)


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
    encrypt_page_images: bool = Field(default=False)
    pipeline_version: str = Field(
        default="1",
        description="Version identifier for ingestion pipeline wiring",
    )

    @model_validator(mode="after")
    def _validate_overlap(self) -> "ProcessingConfig":
        if int(self.chunk_overlap) > int(self.chunk_size):
            raise ValueError("chunk_overlap cannot exceed chunk_size")
        return self


class ChatConfig(BaseModel):
    """Chat memory configuration (ADR-021)."""

    sqlite_path: Path = Field(default=Path("chat.db"))
    memory_store_filter_fetch_cap: int = Field(
        default=5000,
        ge=256,
        le=100000,
        description="Max rows fetched for filtered memory searches before slicing.",
    )
    memory_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for consolidating memory candidates.",
    )
    memory_low_importance_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Importance cutoff for applying memory TTL.",
    )
    memory_low_importance_ttl_days: int = Field(
        default=14,
        ge=0,
        description="TTL in days for low-importance memories (0 disables TTL).",
    )
    memory_max_items_per_namespace: int = Field(
        default=200,
        ge=1,
        description="Maximum memories kept per namespace before eviction.",
    )
    memory_max_candidates_per_turn: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Maximum extracted memory candidates per turn.",
    )


class AgentConfig(BaseModel):
    """Multi-agent system configuration (ADR-011)."""

    enable_multi_agent: bool = Field(default=True)
    decision_timeout: int = Field(default=200, ge=10, le=1000)
    max_retries: int = Field(default=2, ge=0, le=10)
    max_concurrent_agents: int = Field(default=3, ge=1, le=10)
    enable_fallback_rag: bool = Field(default=True)
    use_tool_registry: bool = Field(
        default=True,
        description=(
            "Enable the centralized ToolRegistry (Phase 1 supervisor refactor)."
        ),
    )
    use_shared_llm_client: bool = Field(
        default=True,
        description=(
            "Enable retries for the shared LlamaIndex LLM used by agent workflows "
            "(native retries when available; wrapper fallback otherwise)."
        ),
    )
    enable_deadline_propagation: bool = Field(
        default=False,
        description="Propagate deadlines/cancellation tokens through the graph.",
    )
    enable_router_injection: bool = Field(
        default=False,
        description="Use injected router engines supplied by the tool registry.",
    )

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


class ObservabilityConfig(BaseModel):
    """OpenTelemetry exporter configuration (SPEC-012 / Phase 6)."""

    enabled: bool = Field(
        default=False,
        description="Enable OpenTelemetry tracing and metrics exporters.",
    )
    service_name: str = Field(
        default="docmind-agents",
        description="service.name resource attribute for telemetry exporters.",
    )
    endpoint: str | None = Field(
        default=None,
        description=("Optional OTLP endpoint override for telemetry exporters."),
    )
    protocol: Literal["grpc", "http/protobuf"] = Field(
        default="http/protobuf",
        description="OTLP transport protocol to use for tracing/metrics exporters.",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Additional OTLP headers (for auth tokens, multi-tenant keys).",
    )
    sampling_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling ratio (0 disables tracing entirely).",
    )
    metrics_interval_ms: int = Field(
        default=60_000,
        ge=1_000,
        description="Periodic metrics export interval in milliseconds.",
    )
    instrument_llamaindex: bool = Field(
        default=True,
        description=(
            "Automatically register LlamaIndex OpenTelemetry instrumentation"
            " when exporters are enabled and the integration is installed."
        ),
    )


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
    image_backbone: ImageBackboneName = Field(default="auto")
    siglip_model_id: str = Field(default="google/siglip-base-patch16-224")
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
    enable_image_retrieval: bool = Field(
        default=True,
        description=(
            "Enable visual retrieval (SigLIP text->image) and multimodal fusion."
        ),
    )

    # Server-side fusion settings (ADR-024 v2.8)
    fusion_mode: Literal["rrf", "dbsf"] = Field(
        default="rrf", description="Server-side fusion mode (RRF default)"
    )
    fused_top_k: int = Field(
        default=60, ge=10, le=1000, description="Prefetch/fused candidate cap"
    )
    rrf_k: int = Field(default=60, ge=1, le=256, description="RRF k-constant")
    prefetch_dense_limit: int = Field(
        default=200, ge=1, le=5000, description="Per-branch dense prefetch limit"
    )
    prefetch_sparse_limit: int = Field(
        default=400, ge=1, le=5000, description="Per-branch sparse prefetch limit"
    )
    use_sparse_embeddings: bool = Field(default=True)
    # Deduplication key used before final fused cut
    dedup_key: Literal["page_id", "doc_id"] = Field(default="page_id")
    # Feature flag for future named-vectors multi-head support (no-op now)
    named_vectors_multi_head_enabled: bool = Field(default=False)
    # Router and feature toggles (ADR-003)
    router: Literal["auto", "simple", "hierarchical", "graph"] = Field(default="auto")
    graph_enabled: bool = Field(default=False)
    # Server-side hybrid via Qdrant Query API fusion (prefetch + RRF/DBSF).
    # This specifically controls registration of a server-side hybrid tool in
    # router_factory (distinct from any internal client-side hybrid behavior).
    enable_server_hybrid: bool = Field(
        default=False,
        description=(
            "Enable server-side hybrid retrieval (Qdrant Query API fusion). "
            "Default is False to avoid surprises."
        ),
    )
    # Reranker model
    # (text-only CrossEncoder; ADR-006 legacy, ADR-037 multimodal supersedes)
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    # Visual rerank (SigLIP default; ColPali optional)
    enable_colpali: bool = Field(
        default=False, description="Enable optional ColPali visual reranker"
    )
    siglip_batch_size: int = Field(
        default=8, ge=1, le=64, description="SigLIP image batch size"
    )
    siglip_prune_m: int = Field(
        default=64, ge=1, le=512, description="Pre-fusion prune M for visual rerank"
    )
    # Advanced feature flags (canaries; default to unified behavior)
    device_policy_core: bool = Field(
        default=True,
        description=("Route device/VRAM checks via src.utils.core"),
    )
    siglip_adapter_unified: bool = Field(
        default=True,
        description=("Use shared vision_siglip.load_siglip in adapter"),
    )
    rerank_executor: Literal["thread", "process"] = Field(
        default="thread",
        description=("Executor for rerank timeouts: 'thread' or 'process'"),
    )
    # Optional keyword tool (sparse-only Qdrant text-sparse) registration flag
    # (disabled by default)
    enable_keyword_tool: bool = Field(default=False)

    # --- Centralized reranking timeouts (ms) ---
    # Keep conservative defaults and make all budgets observable in telemetry.
    text_rerank_timeout_ms: int = Field(
        default=250,
        ge=50,
        le=5000,
        description="Timeout (ms) for text cross-encoder reranking stage",
    )
    siglip_timeout_ms: int = Field(
        default=150,
        ge=25,
        le=5000,
        description="Timeout (ms) for SigLIP visual scoring stage",
    )
    colpali_timeout_ms: int = Field(
        default=400,
        ge=25,
        le=10000,
        description="Timeout (ms) for ColPali visual reranking stage",
    )
    total_rerank_budget_ms: int = Field(
        default=800,
        ge=100,
        le=20000,
        description="Overall best-effort budget (ms) across rerank stages",
    )

    # No additional methods; env mapping handled by BaseSettings


class CacheConfig(BaseModel):
    """Document processing cache toggles (ADR-030)."""

    enable_document_caching: bool = Field(default=True)
    ttl_seconds: int = Field(default=3600, ge=300, le=86400)
    max_size_mb: int = Field(default=1000, ge=100, le=10000)
    backend: Literal["duckdb", "sqlite", "memory"] = Field(
        default="duckdb",
        description="Cache backend to use for ingestion artifacts",
    )
    # Path configuration for DuckDB KV store
    dir: Path = Field(default=Path("./cache"))
    filename: str = Field(default="docmind.duckdb")


class ArtifactsConfig(BaseModel):
    """Local content-addressed artifact storage (page images, thumbnails)."""

    dir: Path | None = Field(
        default=None,
        description="Optional override; default is data_dir/artifacts",
    )
    max_total_mb: int = Field(
        default=4096,
        ge=100,
        le=200_000,
        description="Best-effort artifact GC budget (MB)",
    )
    gc_min_age_seconds: int = Field(
        default=3600,
        ge=0,
        le=31_536_000,
        description="Do not GC artifacts newer than this age (seconds)",
    )


class HashingConfig(BaseModel):
    """Deterministic hashing and canonicalisation configuration."""

    canonicalization_version: str = Field(default="1")
    hmac_secret: str = Field(
        default="docmind-dev-secret-please-override-0123456789",
        repr=False,
        description=(
            "Shared secret for HMAC canonical hashes. Override via environment "
            "in production deployments."
        ),
    )
    hmac_secret_version: str = Field(default="1")
    metadata_keys: list[str] = Field(
        default_factory=lambda: [
            "content_type",
            "language",
            "source",
            "source_path",
        ],
        description="Ordered metadata keys included in canonical payloads.",
    )

    @field_validator("hmac_secret")
    @classmethod
    def _validate_hmac_secret(cls, value: str) -> str:
        if len(value.encode("utf-8")) < 32:
            raise ValueError(
                "DOCMIND_HASHING__HMAC_SECRET must be at least 32 bytes "
                "for HMAC strength"
            )
        return value


class DatabaseConfig(BaseModel):
    """Database and vector store configuration."""

    # Vector Database
    vector_store_type: str = Field(default="qdrant")
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="docmind_docs")
    qdrant_image_collection: str = Field(default="docmind_images")
    qdrant_timeout: int = Field(default=60, ge=10, le=300)

    # SQL Database
    sqlite_db_path: Path = Field(default=Path("docmind.db"))
    enable_wal_mode: bool = Field(default=True)


class GraphRAGConfig(BaseModel):
    """GraphRAG configuration (ADR-019)."""

    enabled: bool = Field(default=True)
    relationship_extraction: bool = Field(default=False)
    entity_resolution: Literal["fuzzy", "exact"] = Field(default="fuzzy")
    max_hops: int = Field(default=2, ge=1, le=5)
    max_triplets: int = Field(default=1000, ge=100, le=10000)
    chunk_size: int = Field(default=1024, ge=128, le=8192)
    autoload_policy: Literal["latest_non_stale", "pinned", "ignore"] = Field(
        default="latest_non_stale",
        description="Chat autoload snapshot policy",
    )
    default_path_depth: int = Field(
        default=1, ge=1, le=5, description="Default graph retrieval path depth"
    )
    export_seed_cap: int = Field(
        default=32, ge=1, le=1000, description="Default seed cap for exports"
    )
    pinned_snapshot_id: str | None = Field(
        default=None, description="Pinned snapshot directory name for autoload"
    )


class SnapshotConfig(BaseModel):
    """Snapshot manager configuration."""

    lock_timeout_seconds: float = Field(
        default=10.0, ge=0.5, le=300.0, description="Lock acquisition timeout"
    )
    lock_ttl_seconds: float = Field(
        default=30.0, ge=5.0, le=600.0, description="Lease TTL for metadata"
    )
    retention_count: int = Field(
        default=5, ge=1, le=100, description="Snapshots to retain during GC"
    )
    gc_grace_seconds: int = Field(
        default=86_400,
        ge=0,
        le=604_800,
        description="Grace period before deleting old snapshots",
    )


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
    """Unified DocMind AI configuration with Pydantic Settings V2."""

    model_config = SETTINGS_MODEL_CONFIG

    # Core Application
    app_name: str = Field(default="DocMind AI")
    app_version: str = Field(default="2.0.0")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    # Global cache salt for Streamlit caches; bump to invalidate
    cache_version: int = Field(
        default=0,
        description=(
            "Global Streamlit cache salt. Increment to clear cached data/resources."
        ),
    )

    # File System Paths
    data_dir: Path = Field(default=Path("./data"))
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    cache_dir: Path = Field(default=Path("./cache"))
    log_file: Path = Field(default=Path("./logs/docmind.log"))

    # Canonical hashing (ADR-050, ADR-047)
    hashing: HashingConfig = Field(default_factory=HashingConfig)

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

    # Observability / OpenTelemetry
    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig,
        description="OpenTelemetry exporter configuration.",
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

    # Feature flags
    guided_json_enabled: bool = Field(
        default=False, description="Structured outputs available (SPEC-007 prep)"
    )

    # OpenAI-compatible client configuration group
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)

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
    snapshots: SnapshotConfig = Field(default_factory=SnapshotConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)

    # Alias fields mapped to nested configs for ergonomic top-level
    # environment overrides.
    context_window_size: int | None = Field(default=None)
    chunk_size: int | None = Field(default=None)
    chunk_overlap: int | None = Field(default=None)
    enable_multi_agent: bool | None = Field(default=None)
    # Top-level LLM context window cap (ADR-004/024)
    llm_context_window_max: int = Field(default=131072, ge=8192, le=200000)

    # Image encryption aliases (ADR-031 bridge)
    img_aes_key_base64: str | None = Field(default=None)
    img_kid: str | None = Field(default=None)
    img_delete_plaintext: bool | None = Field(default=None)

    @model_validator(mode="after")
    def _apply_aliases_and_validate(self) -> "DocMindSettings":
        self._apply_alias_overrides()
        self._map_hybrid_to_retrieval()
        self._normalize_persistence_paths()
        self._validate_endpoints_security()
        self._validate_lmstudio_url()
        return self

    def _apply_alias_overrides(self) -> None:
        alias_targets: dict[str, tuple[object, str, Callable[[Any], Any]]] = {
            "model": (self.vllm, "model", str),
            "context_window": (self.vllm, "context_window", int),
            "context_window_size": (self.vllm, "context_window", int),
            "chunk_size": (self.processing, "chunk_size", int),
            "chunk_overlap": (self.processing, "chunk_overlap", int),
            "enable_multi_agent": (self.agents, "enable_multi_agent", bool),
            "img_aes_key_base64": (self.image, "img_aes_key_base64", str),
            "img_kid": (self.image, "img_kid", str),
            "img_delete_plaintext": (self.image, "img_delete_plaintext", bool),
        }
        for field, (target, attr, caster) in alias_targets.items():
            value = getattr(self, field, None)
            if value is None:
                continue
            with suppress(AttributeError, TypeError, ValueError):
                setattr(target, attr, caster(value))

    def _map_hybrid_to_retrieval(self) -> None:
        try:
            fields_set = getattr(self.retrieval, "model_fields_set", set())
            if "enable_server_hybrid" not in fields_set:
                self.retrieval.enable_server_hybrid = bool(self.hybrid.server_side)
            if "rrf_k" not in fields_set:
                self.retrieval.rrf_k = int(self.hybrid.rrf_k)
            if "fusion_mode" not in fields_set:
                self.retrieval.fusion_mode = self.hybrid.method
        except (
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to sync hybrid config into retrieval settings: %s", exc
            )

    def _normalize_persistence_paths(self) -> None:
        """Normalize configured DB paths to live under data_dir by default."""
        for section, attr in (("chat", "sqlite_path"), ("database", "sqlite_db_path")):
            candidate = getattr(getattr(self, section, None), attr, None)
            if not isinstance(candidate, Path):
                continue
            if candidate.is_absolute():
                continue
            if candidate.parent != Path("."):
                # _normalize_persistence_paths: only relocate bare filenames.
                # If `candidate` includes any parent directory
                # (`candidate.parent != Path(".")`), preserve it as-is rather than
                # forcing it under `data_dir`.
                continue
            target = getattr(self, section, None)
            if target is None:
                continue
            setattr(target, attr, self.data_dir / candidate)

    @field_validator("lmstudio_base_url", mode="before")
    @classmethod
    def _norm_lmstudio(cls, v: str) -> str:
        return ensure_v1(v) or v

    @field_validator("llamacpp_base_url", mode="before")
    @classmethod
    def _norm_llamacpp(cls, v: str | None) -> str | None:
        return ensure_v1(v)

    @field_validator("vllm_base_url", mode="before")
    @classmethod
    def _norm_vllm(cls, v: str | None) -> str | None:
        return ensure_v1(v)

    @computed_field
    @property
    def effective_context_window(self) -> int:
        """Return the effective context window with global cap applied."""
        return min(
            int(self.context_window or self.vllm.context_window),
            int(self.llm_context_window_max),
        )

    @computed_field
    @property
    def backend_base_url_normalized(self) -> str | None:
        """Return backend-aware normalized base URL (OpenAI-like -> /v1)."""
        openai_fields_set = getattr(self.openai, "model_fields_set", set())
        openai_base_url = (
            self.openai.base_url
            if "base_url" in openai_fields_set
            and self.openai.base_url
            and self.openai.base_url != _DEFAULT_OPENAI_BASE_URL
            else None
        )
        if self.llm_backend == "ollama":
            return self.ollama_base_url
        if self.llm_backend == "lmstudio":
            # Prefer explicit OpenAI group only when customized
            return ensure_v1(openai_base_url or self.lmstudio_base_url)
        if self.llm_backend == "vllm":
            # Prefer explicit OpenAI-like endpoint only when customized
            base = openai_base_url or self.vllm_base_url or self.vllm.vllm_base_url
            return ensure_v1(base)
        if self.llm_backend == "llamacpp":
            # Prefer explicit OpenAI group only when customized
            return ensure_v1(openai_base_url or self.llamacpp_base_url)
        return None

    def allow_remote_effective(self) -> bool:
        """Return effective allow-remote policy.

        Uses the centralized security settings; no environment overrides.
        """
        return bool(self.security.allow_remote_endpoints)

    def get_vllm_config(self) -> dict[str, Any]:  # pragma: no cover - simple proxy
        """Get vLLM configuration for client setup.

        Returns:
            Mapping with keys ``model``, ``context_window``, and
            ``temperature`` suitable for initializing a vLLM client. Retained
            because launcher utilities import this helper instead of the full
            settings model.
        """
        return {
            "model": self.model or self.vllm.model,
            "context_window": int(self.context_window or self.vllm.context_window),
            "temperature": self.vllm.temperature,
        }

    def get_vllm_env_vars(self) -> dict[str, str]:
        """Return environment variables for vLLM process setup.

        Returns:
            Mapping of environment variable names to string values for vLLM.
            Used by integration scripts that render env files without
            serializing the entire settings object.
        """
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
        """Get model configuration for LlamaIndex setup.

        Returns:
            Mapping with model identifier, effective context window capped by
            ``llm_context_window_max``, generation params, and base URL. The
            LlamaIndex factory consumes this flat dict and predates
            ``model_dump`` usage in containers.
        """
        base_url = self.backend_base_url_normalized
        return {
            "model_name": self.model or self.vllm.model,
            "context_window": self.effective_context_window,
            "max_tokens": self.vllm.max_tokens,
            "temperature": self.vllm.temperature,
            "base_url": base_url,
        }

    def get_embedding_config(self) -> dict[str, Any]:
        """Get embedding configuration for embedding factories.

        Returns:
            Flat mapping used by embedding factory helpers (text+image
            parameters and device selection), while keeping class-based config
            as the single source of truth. Several integration points expect a
            plain dict to hydrate third-party clients.
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
            "trust_remote_code": bool(self.security.trust_remote_code),
        }

    def get_graphrag_config(self) -> dict[str, Any]:
        """Get GraphRAG configuration for ADR-019.

        Returns:
            Mapping with GraphRAG enablement and extraction/traversal settings.
            Router initialization consumes this helper directly.
        """
        c = self.graphrag_cfg
        return {
            "enabled": self.is_graphrag_enabled(),
            "relationship_extraction": c.relationship_extraction,
            "entity_resolution": c.entity_resolution,
            "max_hops": c.max_hops,
            "max_triplets": c.max_triplets,
            "chunk_size": c.chunk_size,
        }

    def is_graphrag_enabled(self) -> bool:
        """Return True when both the global and nested GraphRAG flags allow it."""
        base_flag = bool(getattr(self, "enable_graphrag", False))
        if not base_flag:
            return False
        try:
            graphrag_cfg = getattr(self, "graphrag_cfg", None)
        except (AttributeError, TypeError):
            return base_flag
        if graphrag_cfg is None:
            return base_flag
        try:
            return base_flag and bool(getattr(graphrag_cfg, "enabled", True))
        except (AttributeError, TypeError, ValueError):
            return base_flag

    # === Validation helpers ===
    def _validate_endpoints_security(self) -> None:
        """Validate endpoint URLs against security policy.

        When ``allow_remote_endpoints`` is False, only localhost/127.0.0.1
        URLs are permitted. Users may extend ``endpoint_allowlist``.

        Raises:
            ValueError: If any configured base URL is not allowed while
            ``allow_remote_endpoints`` is False.
        """
        if self.security.allow_remote_endpoints:
            return

        def _normalize_host(host: str) -> str:
            return (host or "").strip().lower().rstrip(".")

        def _is_allowed(url: str | None) -> bool:
            if not url:
                return True
            try:
                parsed = urlparse(url)
                # If malformed, treat as not allowed (defensive)
                if not parsed.scheme or not parsed.netloc:
                    return False

                host = _normalize_host(parsed.hostname or "")
                # Always accept explicit loopback hosts
                if host in {"localhost", "127.0.0.1", "::1"}:
                    return True

                # Build a set of allowed hostnames from the allowlist entries
                allowed_hosts: set[str] = set()
                for entry in self.security.endpoint_allowlist:
                    e = (entry or "").strip()
                    if not e:
                        continue
                    ep = urlparse(e)
                    if ep.hostname:
                        entry_host = _normalize_host(ep.hostname)
                    else:
                        # Fallback: if an entry is just a hostname
                        entry_host = _normalize_host(e.split("/")[0].split(":")[0])
                    allowed_hosts.add(entry_host)

                return host in allowed_hosts
            except (ValueError, TypeError):  # pragma: no cover - defensive
                return False

        raw_urls = {
            self.ollama_base_url,
            self.openai.base_url,
            self.lmstudio_base_url,
            self.vllm_base_url,
            getattr(self.vllm, "vllm_base_url", None),
            self.llamacpp_base_url,
        }
        for url in raw_urls:
            if not _is_allowed(ensure_v1(url)):
                raise ValueError(
                    "Remote endpoints are disabled. Set allow_remote_endpoints=True "
                    "or use localhost URLs."
                )

    def _validate_lmstudio_url(self) -> None:
        """Ensure LM Studio base URL ends with ``/v1``.

        Raises:
            ValueError: If ``lmstudio_base_url`` is set and does not end with
            ``/v1`` as required by the API.
        """
        if self.lmstudio_base_url and not self.lmstudio_base_url.rstrip("/").endswith(
            "/v1"
        ):
            raise ValueError("LM Studio base URL must end with /v1")


# Global settings instance - primary interface for the application
settings = DocMindSettings()

# Startup side-effects (logging, env bridges) are handled in startup_init()
# located in src.config.integrations.

# Module exports
__all__ = [
    "AgentConfig",
    "AnalysisConfig",
    "ArtifactsConfig",
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
