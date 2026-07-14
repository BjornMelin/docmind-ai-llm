"""Unified DocMind AI configuration using Pydantic Settings v2.

Provides a typed, nested configuration model with environment variable
mapping. Prefer nested fields and `DOCMIND_{SECTION}__{FIELD}` env vars.

Usage:
    from src.config.settings import settings
    print(settings.embedding.model_name)
"""

import base64
import os
import re
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from src.config.dotenv import resolve_dotenv_path
from src.config.embedding_defaults import (
    BGE_M3_EMBEDDING_DIMENSION,
    DEFAULT_BGE_M3_MODEL_ID,
    DEFAULT_BGE_M3_MODEL_REVISION,
    DEFAULT_BGE_RERANKER_MODEL_ID,
)
from src.config.settings_utils import (
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_VLLM_BASE_URL,
    endpoint_url_allowed,
    ensure_http_scheme,
    ensure_v1,
    parse_endpoint_allowlist_hosts,
)
from src.nlp.settings import SpacyNlpSettings
from src.version import get_version

SETTINGS_MODEL_CONFIG = SettingsConfigDict(
    # Keep dotenv loading opt-in (callers can pass `_env_file=...` or use
    # `bootstrap_settings`). Using a non-existent default path keeps the default
    # behavior "no implicit dotenv" while still allowing `_env_file` overrides.
    env_file=".env.disabled",
    env_prefix="DOCMIND_",
    env_nested_delimiter="__",
    case_sensitive=False,
    extra="ignore",
    populate_by_name=True,
)

DotenvPriorityMode = Literal["env_first", "dotenv_first"]

# Bootstrapped mode for this process. When unset, defaults to env-first.
_DOTENV_PRIORITY_MODE: DotenvPriorityMode | None = None

_CONFIG_ENV_PREFIX = "DOCMIND_CONFIG__"
_VALID_ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two dictionaries where `override` wins.

    Args:
        base: Base dictionary.
        override: Override dictionary. Values in this mapping take precedence.

    Returns:
        A new merged dictionary.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(cast(dict[str, Any], merged[key]), value)
        else:
            merged[key] = value
    return merged


def _get_dotenv_priority_mode() -> DotenvPriorityMode:
    """Return the effective dotenv priority mode.

    Precedence:
    1) `DOCMIND_CONFIG__DOTENV_PRIORITY` from process env
    2) `_DOTENV_PRIORITY_MODE` (used by tests)
    3) `"env_first"` default

    Returns:
        The effective dotenv priority mode.
    """
    # Intentional direct env read for bootstrap configuration. This runs
    # before Pydantic settings are constructed and is a narrow exception to
    # the general "no direct os.getenv/os.environ sprawl" policy.
    raw = os.environ.get(f"{_CONFIG_ENV_PREFIX}DOTENV_PRIORITY")
    if raw:
        value = raw.strip().lower()
        if value in {"env_first", "dotenv_first"}:
            return cast(DotenvPriorityMode, value)
    if _DOTENV_PRIORITY_MODE is not None:
        return _DOTENV_PRIORITY_MODE
    return "env_first"


def _parse_csv(value: str | None) -> list[str]:
    """Parse a comma-separated string into trimmed, non-empty items.

    Args:
        value: Raw comma-separated string.

    Returns:
        List of trimmed, non-empty items.
    """
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _validate_env_key_name(key: str) -> None:
    """Validate env var key names for bootstrap toggles.

    Args:
        key: Environment variable name to validate.

    Raises:
        ValueError: If `key` is not `[A-Z][A-Z0-9_]*`.
    """
    if not key or _VALID_ENV_KEY_RE.fullmatch(key) is None:
        raise ValueError("Invalid env var key; expected [A-Z][A-Z0-9_]*")


def _validate_env_value_no_controls(value: str) -> None:
    """Reject env var values containing ASCII control characters.

    Args:
        value: Environment variable value.

    Raises:
        ValueError: If `value` contains ASCII control characters.
    """
    if re.search(r"[\x00-\x1f\x7f]", value):
        raise ValueError("Env var value contains control characters")


@dataclass(frozen=True, slots=True)
class _BootstrapOptions:
    dotenv_priority: DotenvPriorityMode | None
    env_mask_keys: list[str]
    env_overlays: list[tuple[str, str]]


def _read_bootstrap_options(dotenv_path: Path) -> _BootstrapOptions:
    """Read DOCMIND_CONFIG__* bootstrap toggles from env and/or `.env`.

    Args:
        dotenv_path: Path to the repo dotenv file.

    Returns:
        Parsed bootstrap options.

    Raises:
        ValueError: For malformed toggle values (invalid keys, bad overlay specs).
    """
    dotenv_kv: dict[str, str | None] = {}
    try:
        from dotenv import dotenv_values as _dotenv_values

        dotenv_kv = cast(
            dict[str, str | None],
            _dotenv_values(dotenv_path, encoding="utf-8", interpolate=False),
        )
    except Exception:
        dotenv_kv = {}

    def _cfg(name: str) -> str | None:
        key = f"{_CONFIG_ENV_PREFIX}{name}"
        if key in os.environ:
            return os.environ[key]
        return dotenv_kv.get(key)

    raw_priority = _cfg("DOTENV_PRIORITY")
    if raw_priority is None:
        dotenv_priority = None
    else:
        value = raw_priority.strip().lower()
        if value not in {"env_first", "dotenv_first"}:
            raise ValueError(
                "DOCMIND_CONFIG__DOTENV_PRIORITY must be 'env_first' or 'dotenv_first'"
            )
        dotenv_priority = cast(DotenvPriorityMode, value)

    env_mask_keys: list[str] = []
    for key in _parse_csv(_cfg("ENV_MASK_KEYS")):
        _validate_env_key_name(key)
        if key.startswith(_CONFIG_ENV_PREFIX):
            raise ValueError("Refusing to mask DOCMIND_CONFIG__* keys")
        env_mask_keys.append(key)

    overlays: list[tuple[str, str]] = []
    for spec in _parse_csv(_cfg("ENV_OVERLAY")):
        if ":" not in spec:
            raise ValueError(
                "DOCMIND_CONFIG__ENV_OVERLAY entries must be KEY:settings.path"
            )
        key, path = spec.split(":", 1)
        key = key.strip()
        path = path.strip()
        _validate_env_key_name(key)
        if not path or any(part.strip() == "" for part in path.split(".")):
            raise ValueError(f"Invalid settings path for overlay key={key}")
        overlays.append((key, path))

    return _BootstrapOptions(
        dotenv_priority=dotenv_priority,
        env_mask_keys=env_mask_keys,
        env_overlays=overlays,
    )


def _apply_env_mask(keys: list[str]) -> None:
    """Delete selected env vars from `os.environ` (best-effort).

    Args:
        keys: Environment variable names to delete.
    """
    for key in keys:
        with suppress(KeyError):
            del os.environ[key]


def _apply_env_overlay(overlays: list[tuple[str, str]]) -> None:
    """Materialize overlays from loaded settings into `os.environ`.

    Each overlay is `(ENV_KEY, settings.path)`. Values are extracted from the
    already-loaded `settings` instance.

    Args:
        overlays: Overlay specs.

    Raises:
        ValueError: If a referenced settings path is unknown.
    """
    for key, path in overlays:
        current: Any = settings
        for part in path.split("."):
            if not hasattr(current, part):
                raise ValueError(f"Unknown settings path for overlay key={key}: {path}")
            current = getattr(current, part)

        if isinstance(current, SecretStr):
            value = current.get_secret_value()
        elif current is None:
            value = ""
        else:
            value = str(current)

        if not value:
            with suppress(KeyError):
                del os.environ[key]
            continue
        _validate_env_value_no_controls(value)
        os.environ[key] = value


class LLMRequestConfig(BaseModel):
    """Provider-neutral model and generation request controls."""

    model: str | None = Field(
        default=None,
        description="Optional model override for the selected provider.",
    )
    context_window: int = Field(default=131072, ge=8192, le=200000)
    max_output_tokens: int = Field(default=2048, ge=100, le=8192)
    temperature: float = Field(default=0.1, ge=0, le=2)


class OpenAIConfig(BaseModel):
    """Settings for OpenAI-compatible servers.

    Used for OpenAI-compatible endpoints (local servers, proxies, or cloud gateways).
    Base URL normalization (optional `/v1`) is applied at selection time via
    :meth:`DocMindSettings.backend_base_url_normalized`.
    """

    model_config = ConfigDict(validate_assignment=True)

    base_url: AnyHttpUrl = Field(default=DEFAULT_OPENAI_BASE_URL)
    api_key: SecretStr | None = Field(
        default=None, description="Optional API key", repr=False
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional HTTP headers for OpenAI-compatible providers "
            "(e.g., OpenRouter HTTP-Referer/X-Title)."
        ),
    )
    require_v1: bool = Field(
        default=True,
        description=(
            "When True, normalize base_url to include a single '/v1' suffix. "
            "Disable only for OpenAI-compatible endpoints rooted at '/' "
            "(e.g., LiteLLM Proxy default)."
        ),
    )
    api_mode: Literal["chat_completions", "responses"] = Field(
        default="chat_completions",
        description="Select OpenAI-compatible API mode (legacy chat vs /responses).",
    )

    @field_validator("base_url", mode="before")
    @classmethod
    def _ensure_scheme_on_base(cls, v: object) -> str:
        candidate = ensure_http_scheme(v) or ""
        return candidate

    @field_validator("default_headers", mode="after")
    @classmethod
    def _normalize_default_headers(
        cls, v: dict[str, str] | None
    ) -> dict[str, str] | None:
        if v is None:
            return None
        out: dict[str, str] = {}
        for raw_k, raw_val in v.items():
            k = str(raw_k).strip()
            if not k:
                continue
            val = str(raw_val).strip()
            if not val:
                continue
            if re.search(r"[\x00-\x1f\x7f]", k) or re.search(r"[\x00-\x1f\x7f]", val):
                raise ValueError(
                    "DOCMIND_OPENAI__DEFAULT_HEADERS may not contain control "
                    "characters or newlines"
                )
            out[k] = val
        return out or None


class ImageEncryptionConfig(BaseModel):
    """AES-GCM image encryption settings (AES-256 key; optional)."""

    model_config = ConfigDict(validate_assignment=True)

    aes_key_base64: SecretStr | None = Field(default=None, repr=False)
    kid: str | None = Field(default=None)
    delete_plaintext: bool = Field(default=False)

    @field_validator("aes_key_base64", mode="before")
    @classmethod
    def _validate_aes_key_base64(cls, v: object) -> str | None:
        if v is None:
            return None
        if isinstance(v, SecretStr):
            v = v.get_secret_value()
        raw = str(v).strip()
        if not raw:
            return None
        try:
            decoded = base64.b64decode(raw, validate=True)
        except Exception as exc:
            raise ValueError("DOCMIND_IMG_AES_KEY_BASE64 must be valid base64") from exc
        if len(decoded) != 32:
            raise ValueError(
                "DOCMIND_IMG_AES_KEY_BASE64 must decode to 32 bytes (AES-256)"
            )
        return raw


class TelemetryConfig(BaseModel):
    """Local-first JSONL telemetry controls (ADR-032)."""

    model_config = ConfigDict(validate_assignment=True)

    disabled: bool = Field(default=False)
    sample: float = Field(default=1.0, ge=0.0, le=1.0)
    rotate_bytes: int = Field(default=0, ge=0)
    jsonl_path: Path = Field(default=Path("./logs/telemetry.jsonl"))


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


class ProcessingConfig(BaseModel):
    """Document processing configuration (ADR-009)."""

    chunk_size: int = Field(default=1500, ge=100, le=10000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)
    max_document_size_mb: int = Field(default=100, ge=1, le=500)
    encrypt_page_images: bool = Field(default=False)

    @model_validator(mode="after")
    def _validate_overlap(self) -> "ProcessingConfig":
        if int(self.chunk_overlap) > int(self.chunk_size):
            raise ValueError("chunk_overlap cannot exceed chunk_size")
        return self


class DocumentParsingConfig(BaseModel):
    """CPU-safe document parser routing and resource limits."""

    framework: Literal["docling"] = Field(default="docling")
    profile: Literal["cpu_safe"] = Field(default="cpu_safe")
    model_cache_dir: Path = Field(default=Path("./cache/models"))
    max_pages: int = Field(default=500, ge=1, le=5000)
    max_render_pixels: int = Field(default=40_000_000, ge=1_000_000, le=100_000_000)
    max_total_text_chars: int = Field(
        default=10_000_000,
        ge=10_000,
        le=100_000_000,
    )
    parse_timeout_seconds: float = Field(default=300.0, ge=1.0, le=1800.0)
    ocrmypdf_timeout_seconds: float = Field(default=300.0, ge=1.0, le=1800.0)
    direct_text_probe_bytes: int = Field(default=8192, ge=512, le=65_536)

    @field_validator("model_cache_dir", mode="before")
    @classmethod
    def _reject_blank_model_cache_dir(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("model_cache_dir must not be empty")
        return value


class PdfBackendConfig(BaseModel):
    """PDF inspection and rasterization backend configuration."""

    render_dpi: int = Field(default=200, ge=72, le=600)
    min_text_chars_per_page: int = Field(default=24, ge=0, le=10000)


class OcrConfig(BaseModel):
    """RapidOCR and optional searchable-PDF configuration."""

    engine: Literal["rapidocr"] = Field(default="rapidocr")
    force_ocr: bool = Field(default=False)
    searchable_pdf_enabled: bool = Field(default=False)
    ocrmypdf_jobs: int = Field(default=1, ge=1, le=8)


class ChatConfig(BaseModel):
    """Chat memory configuration (ADR-021)."""

    sqlite_path: Path = Field(default=Path("chat.db"))
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
        description=(
            "Maximum memories per namespace. Automatic consolidation evicts only "
            "older derived memories; explicit writes fail when no safe capacity "
            "remains."
        ),
    )
    memory_max_candidates_per_turn: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Maximum extracted memory candidates per turn.",
    )


class AgentConfig(BaseModel):
    """Multi-agent system configuration (ADR-011)."""

    decision_timeout: int = Field(default=200, ge=10, le=1000)
    max_retries: int = Field(default=2, ge=0, le=10)


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

    Text uses BGE-M3. Images use SigLIP.
    """

    # Text (BGE-M3)
    model_name: str = Field(default=DEFAULT_BGE_M3_MODEL_ID)
    model_revision: str | None = Field(
        default=DEFAULT_BGE_M3_MODEL_REVISION,
        description=(
            "Pinned Hugging Face revision. Custom model IDs remain unpinned "
            "unless this field is set explicitly."
        ),
    )
    local_model_path: Path | None = Field(
        default=None,
        description="Optional local SentenceTransformers snapshot directory.",
    )
    cache_folder: Path = Field(default=Path("./models_cache"))
    dimension: int = Field(default=BGE_M3_EMBEDDING_DIMENSION, ge=256, le=4096)
    max_length: int = Field(default=8192, ge=512, le=16384)
    normalize_text: bool = Field(default=True)
    batch_size_text_gpu: int = Field(default=12, ge=1, le=128)
    batch_size_text_cpu: int = Field(default=4, ge=1, le=64)

    @field_validator("model_revision", mode="before")
    @classmethod
    def _normalize_model_revision(cls, value: Any) -> str | None:
        """Normalize blank text-model revisions to `None`."""
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("local_model_path", mode="before")
    @classmethod
    def _normalize_local_model_path(cls, value: Any) -> Any:
        """Treat a blank local model path as unset."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _validate_text_model_contract(self) -> "EmbeddingConfig":
        """Keep the canonical BGE-M3 model and Qdrant dimension aligned."""
        if self.model_name == DEFAULT_BGE_M3_MODEL_ID:
            if self.dimension != BGE_M3_EMBEDDING_DIMENSION:
                raise ValueError(
                    "BAAI/bge-m3 requires embedding dimension "
                    f"{BGE_M3_EMBEDDING_DIMENSION}"
                )
            if self.model_revision is None:
                self.model_revision = DEFAULT_BGE_M3_MODEL_REVISION
        elif "model_revision" not in self.model_fields_set:
            self.model_revision = None
        return self

    # Images
    siglip_model_id: str = Field(default="google/siglip-base-patch16-224")
    siglip_model_revision: str | None = Field(
        default=None,
        description=(
            "Optional pinned Hugging Face revision for SigLIP. When unset, the "
            "default SigLIP model uses the repo-curated revision pin; custom "
            "model IDs load without a revision unless explicitly configured."
        ),
    )

    @field_validator("siglip_model_revision", mode="before")
    @classmethod
    def _normalize_siglip_model_revision(
        cls,
        value: Any,
    ) -> str | None:
        """Normalize blank SigLIP revisions to `None`."""
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    normalize_image: bool = Field(default=True)
    batch_size_image: int = Field(default=8, ge=1, le=64)


class RetrievalConfig(BaseModel):
    """Retrieval and reranking configuration (ADR-006)."""

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
    # Deduplication key used before final fused cut
    dedup_key: Literal["page_id", "doc_id"] = Field(default="page_id")
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
    reranker_model: str = Field(default=DEFAULT_BGE_RERANKER_MODEL_ID)
    # Visual rerank
    siglip_batch_size: int = Field(
        default=8, ge=1, le=64, description="SigLIP image batch size"
    )
    siglip_prune_m: int = Field(
        default=64, ge=1, le=512, description="Pre-fusion prune M for visual rerank"
    )
    enable_keyword_tool: bool = Field(
        default=False,
        description="Expose the sparse-only keyword tool to the retrieval agent.",
    )
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
    total_rerank_budget_ms: int = Field(
        default=400,
        ge=100,
        le=20000,
        description="Overall best-effort budget (ms) across rerank stages",
    )

    # No additional methods; env mapping handled by BaseSettings


class CacheConfig(BaseModel):
    """Document processing cache path configuration (ADR-030)."""

    dir: Path = Field(default=Path("./cache"))
    filename: str = Field(default="docmind.duckdb")

    @property
    def ingestion_db_path(self) -> Path:
        """Return the canonical live ingestion DuckDB path.

        Returns:
            Path: Cache directory, ingestion subdirectory, and configured filename.
        """
        return self.dir / "ingestion" / self.filename


class ArtifactsConfig(BaseModel):
    """Local content-addressed artifact storage (page images, thumbnails)."""

    dir: Path | None = Field(
        default=None,
        description="Optional override; default is data_dir/artifacts",
    )


class HashingConfig(BaseModel):
    """Deterministic hashing and canonicalisation configuration."""

    canonicalization_version: str = Field(default="1")
    hmac_secret: SecretStr = Field(
        default=SecretStr("docmind-dev-secret-please-override-0123456789"),
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
    def _validate_hmac_secret(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value().encode("utf-8")) < 32:
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
    qdrant_api_key: SecretStr | None = Field(
        default=None,
        repr=False,
        description="Optional API key for authenticated Qdrant endpoints.",
    )
    qdrant_collection: str = Field(default="docmind_docs")
    qdrant_image_collection: str = Field(default="docmind_images")
    qdrant_timeout: int = Field(default=60, ge=10, le=300)

    @field_validator("qdrant_collection", "qdrant_image_collection")
    @classmethod
    def _validate_collection_name(cls, value: str) -> str:
        """Require a non-empty Qdrant owner that is safe as a local path segment."""
        candidate = value.strip()
        if (
            not candidate
            or candidate in {".", ".."}
            or "/" in candidate
            or "\\" in candidate
            or any(ord(char) < 32 for char in candidate)
        ):
            raise ValueError("Qdrant collection names must be safe non-empty names")
        return candidate


class GraphRAGConfig(BaseModel):
    """GraphRAG configuration (ADR-019)."""

    enabled: bool = Field(default=False)
    autoload_policy: Literal["latest_non_stale", "ignore"] = Field(
        default="latest_non_stale",
        description="Chat autoload snapshot policy",
    )
    default_path_depth: int = Field(
        default=1, ge=1, le=5, description="Default graph retrieval path depth"
    )
    export_seed_cap: int = Field(
        default=32, ge=1, le=1000, description="Default seed cap for exports"
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

    progress_poll_interval_sec: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Polling interval for background job progress updates.",
    )

    request_timeout_seconds: int = Field(default=30, ge=5, le=300)


class DocMindSettings(BaseSettings):
    """Unified DocMind AI configuration with Pydantic Settings V2."""

    model_config = SETTINGS_MODEL_CONFIG

    # Core Application
    app_name: str = Field(default="DocMind AI")
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
    log_file: Path = Field(default=Path("./logs/docmind.log"))

    # Canonical hashing (ADR-050, ADR-047)
    hashing: HashingConfig = Field(default_factory=HashingConfig)

    # Analytics (ADR-032)
    analytics_enabled: bool = Field(
        default=False, description="Enable optional local DuckDB analytics database"
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
    llm_backend: Literal[
        "vllm",
        "ollama",
        "lmstudio",
        "llamacpp",
        "openai_compatible",
    ] = Field(default="ollama")
    ollama_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Optional Ollama Cloud API key (Bearer token). When set, it is used to "
            "authenticate to https://ollama.com for cloud access and web search."
        ),
    )
    ollama_enable_web_search: bool = Field(
        default=False,
        description=(
            "Enable Ollama Cloud web_search/web_fetch tools (requires API key and "
            "remote endpoints allowed)."
        ),
    )
    ollama_enable_logprobs: bool = Field(
        default=False,
        description="Enable token logprobs for Ollama chat (default: off).",
    )
    ollama_top_logprobs: int = Field(
        default=0,
        ge=0,
        le=20,
        description=(
            "Number of alternative tokens per position to include when logprobs are "
            "enabled (0-20)."
        ),
    )
    ollama_base_url: AnyHttpUrl = Field(
        default=DEFAULT_OLLAMA_BASE_URL,
    )
    lmstudio_base_url: AnyHttpUrl = Field(default=DEFAULT_LMSTUDIO_BASE_URL)
    vllm_base_url: AnyHttpUrl = Field(
        default=DEFAULT_VLLM_BASE_URL,
        description="vLLM OpenAI-compatible HTTP endpoint",
    )
    llamacpp_base_url: AnyHttpUrl | None = Field(
        default=None, description="Optional llama.cpp server (OpenAI-compatible)"
    )
    enable_gpu_acceleration: bool = Field(default=True)

    # Environment / telemetry
    environment: str | None = Field(
        default=None,
        description=(
            "Optional environment name (e.g., dev/staging/prod). When set, it is "
            "used for telemetry/resource tagging (DOCMIND_ENVIRONMENT)."
        ),
    )
    telemetry: TelemetryConfig = Field(
        default_factory=TelemetryConfig,
        description="Local JSONL telemetry configuration.",
    )

    # OpenAI-compatible client configuration group
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)

    # Global LLM client behavior
    llm_request_timeout_seconds: int = Field(default=120, ge=5, le=600)
    llm_streaming_enabled: bool = Field(default=True)

    # Advanced Features
    # UI Configuration moved to nested UIConfig structure

    # Nested Configuration Models
    llm_request: LLMRequestConfig = Field(default_factory=LLMRequestConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    spacy: SpacyNlpSettings = Field(default_factory=SpacyNlpSettings)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    # Additional nested configs (ADR-024)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    graphrag_cfg: GraphRAGConfig = Field(default_factory=GraphRAGConfig)
    snapshots: SnapshotConfig = Field(default_factory=SnapshotConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    parsing: DocumentParsingConfig = Field(default_factory=DocumentParsingConfig)
    pdf_backend: PdfBackendConfig = Field(default_factory=PdfBackendConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    image_encryption: ImageEncryptionConfig = Field(
        default_factory=ImageEncryptionConfig
    )
    # Flat env bridges (SPEC-031): keep existing flat env vars without
    # introducing new nested namespaces.
    telemetry_disabled: bool | None = Field(default=None, exclude=True)
    telemetry_sample: float | None = Field(default=None, exclude=True)
    telemetry_rotate_bytes: int | None = Field(default=None, exclude=True)
    img_aes_key_base64: str | None = Field(default=None, repr=False, exclude=True)
    img_kid: str | None = Field(default=None, exclude=True)
    img_delete_plaintext: bool | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _apply_aliases_and_validate(self) -> "DocMindSettings":
        self._apply_alias_overrides()
        self._normalize_persistence_paths()
        self._validate_endpoints_security()
        self._validate_lmstudio_url()
        self._validate_web_search_config()
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize precedence, optionally preferring repo `.env` over env vars.

        Default remains: init kwargs > environment > dotenv > secrets.

        When `DOCMIND_CONFIG__DOTENV_PRIORITY=dotenv_first`, `.env` wins over
        environment variables for DocMind settings, except for the `security`
        subtree where environment variables remain higher precedence as a safety
        guard.
        """
        if _get_dotenv_priority_mode() != "dotenv_first":
            return init_settings, env_settings, dotenv_settings, file_secret_settings

        def combined_env_dotenv() -> dict[str, Any]:
            env_data = cast(dict[str, Any], env_settings())
            dotenv_data = cast(dict[str, Any], dotenv_settings())
            merged = _deep_merge(env_data, dotenv_data)  # dotenv wins by default

            # Guardrail: security config is env-first even when dotenv-first.
            if "security" in env_data:
                merged_security = merged.get("security", {})
                if isinstance(merged_security, dict) and isinstance(
                    env_data["security"], dict
                ):
                    merged["security"] = _deep_merge(
                        cast(dict[str, Any], merged_security),
                        cast(dict[str, Any], env_data["security"]),
                    )
                else:
                    merged["security"] = env_data["security"]
            return merged

        return (
            init_settings,
            cast(PydanticBaseSettingsSource, combined_env_dotenv),
            file_secret_settings,
        )

    def _apply_alias_overrides(self) -> None:
        alias_targets: dict[str, tuple[object, str, Callable[[Any], Any]]] = {
            # Flat env vars bridged into nested configs.
            "telemetry_disabled": (self.telemetry, "disabled", bool),
            "telemetry_sample": (self.telemetry, "sample", float),
            "telemetry_rotate_bytes": (self.telemetry, "rotate_bytes", int),
            "img_aes_key_base64": (
                self.image_encryption,
                "aes_key_base64",
                lambda x: x,
            ),
            "img_kid": (self.image_encryption, "kid", str),
            "img_delete_plaintext": (self.image_encryption, "delete_plaintext", bool),
        }
        for field, (target, attr, caster) in alias_targets.items():
            value = getattr(self, field, None)
            if value is None:
                continue
            if isinstance(value, SecretStr):
                value = value.get_secret_value()
            setattr(target, attr, caster(value))

    def _normalize_persistence_paths(self) -> None:
        """Move a bare chat database filename under ``data_dir``."""
        candidate = self.chat.sqlite_path
        if candidate.is_absolute() or candidate.parent != Path("."):
            return
        self.chat.sqlite_path = self.data_dir / candidate

    @field_validator("lmstudio_base_url", mode="before")
    @classmethod
    def _norm_lmstudio(cls, v: object) -> str:
        candidate = ensure_http_scheme(v) or ""
        return ensure_v1(candidate) or candidate

    @field_validator("llamacpp_base_url", mode="before")
    @classmethod
    def _norm_llamacpp(cls, v: object | None) -> str | None:
        if v is None:
            return None
        candidate = ensure_http_scheme(v) or ""
        return ensure_v1(candidate) or candidate or None

    @field_validator("vllm_base_url", mode="before")
    @classmethod
    def _norm_vllm(cls, v: object) -> str:
        candidate = ensure_http_scheme(v) or ""
        return ensure_v1(candidate) or candidate

    @field_validator("ollama_base_url", mode="before")
    @classmethod
    def _norm_ollama(cls, v: object) -> str:
        candidate = ensure_http_scheme(v) or ""
        return candidate

    @computed_field
    @property
    def effective_model(self) -> str:
        """Return the backend-aware model identifier.

        Returns:
            str: Explicit model override or the selected backend's default model.
        """
        if self.llm_request.model:
            return self.llm_request.model
        if self.llm_backend == "ollama":
            return "qwen3:4b-instruct"
        return "Qwen/Qwen3-4B-Instruct-2507-FP8"

    @computed_field
    @property
    def effective_context_window(self) -> int:
        """Return the provider-neutral request context window."""
        return int(self.llm_request.context_window)

    @computed_field
    @property
    def backend_base_url_normalized(self) -> str | None:
        """Return backend-aware base URL.

        For OpenAI-compatible endpoints, `/v1` normalization is applied when
        `openai.require_v1` is enabled.
        """
        if self.llm_backend == "ollama":
            return str(self.ollama_base_url).rstrip("/")
        if self.llm_backend == "openai_compatible":
            raw = str(self.openai.base_url).rstrip("/")
            return ensure_v1(raw) if self.openai.require_v1 else raw
        if self.llm_backend == "lmstudio":
            return ensure_v1(self.lmstudio_base_url)
        if self.llm_backend == "vllm":
            return ensure_v1(self.vllm_base_url)
        if self.llm_backend == "llamacpp":
            return ensure_v1(self.llamacpp_base_url)
        return None

    def allow_remote_effective(self) -> bool:
        """Return effective allow-remote policy.

        Uses the centralized security settings; no environment overrides.
        """
        return bool(self.security.allow_remote_endpoints)

    def get_embedding_config(self) -> dict[str, Any]:
        """Get embedding configuration for embedding factories.

        Returns:
            Flat mapping used by embedding factory helpers (text+image
            parameters and device selection), while keeping class-based config
            as the single source of truth. Several integration points expect a
            plain dict to hydrate third-party clients.
        """
        from src.utils.core import select_device

        device = select_device("auto") if self.enable_gpu_acceleration else "cpu"
        local_model_path = self.embedding.local_model_path
        model_name = (
            str(local_model_path.expanduser())
            if local_model_path is not None
            else self.embedding.model_name
        )
        return {
            # Text
            "model_id": self.embedding.model_name,
            "model_name": model_name,
            "model_revision": (
                None if local_model_path is not None else self.embedding.model_revision
            ),
            "local_model_path": (
                str(local_model_path.expanduser())
                if local_model_path is not None
                else None
            ),
            "cache_folder": str(self.embedding.cache_folder.expanduser()),
            "local_files_only": True,
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
            # Images
            "siglip_model_id": self.embedding.siglip_model_id,
            "siglip_model_revision": self.embedding.siglip_model_revision,
            "batch_size_image": self.embedding.batch_size_image,
            "normalize_image": self.embedding.normalize_image,
            "trust_remote_code": bool(self.security.trust_remote_code),
        }

    # === Validation helpers ===
    def _validate_endpoints_security(self) -> None:
        """Validate endpoint URLs against security policy.

        When ``allow_remote_endpoints`` is False:
        - Loopback hosts are always allowed.
        - Non-loopback hosts must be explicitly allowlisted.
        - Allowlisted hostnames are DNS-resolved and rejected if they map to
          private/link-local/reserved ranges (defense-in-depth against SSRF and
          DNS rebinding).

        Raises:
            ValueError: If any configured base URL is not allowed while
            ``allow_remote_endpoints`` is False.
        """
        if self.security.allow_remote_endpoints:
            return

        allowed_hosts = parse_endpoint_allowlist_hosts(self.security.endpoint_allowlist)

        raw_urls = {
            self.ollama_base_url,
            self.openai.base_url,
            self.lmstudio_base_url,
            self.vllm_base_url,
            self.llamacpp_base_url,
        }
        for url in raw_urls:
            normalized = ensure_v1(url) if url is not None else None
            if not endpoint_url_allowed(normalized, allowed_hosts=allowed_hosts):
                raise ValueError(
                    "Remote endpoints are disabled. Set allow_remote_endpoints=True "
                    "or use localhost URLs."
                )
        if not endpoint_url_allowed(
            self.database.qdrant_url,
            allowed_hosts=allowed_hosts,
        ):
            raise ValueError(
                "Remote endpoints are disabled. Set allow_remote_endpoints=True "
                "or use a localhost Qdrant URL."
            )

    def _validate_lmstudio_url(self) -> None:
        """Ensure LM Studio base URL ends with ``/v1``.

        Raises:
            ValueError: If ``lmstudio_base_url`` is set and does not end with
            ``/v1`` as required by the API.
        """
        if self.lmstudio_base_url and not str(self.lmstudio_base_url).rstrip(
            "/"
        ).endswith("/v1"):
            raise ValueError("LM Studio base URL must end with /v1")

    def _validate_web_search_config(self) -> None:
        """Validate Ollama web search prerequisites.

        Web search requires an API key and permission to reach Ollama Cloud.
        Permission may be global or limited to an explicitly allowlisted
        ``ollama.com`` endpoint that passes the canonical DNS/IP policy.

        Raises:
            ValueError: If web search is enabled without required prerequisites.
        """
        if not self.ollama_enable_web_search:
            return

        if self.ollama_api_key is None or not self.ollama_api_key.get_secret_value():
            raise ValueError(
                "ollama_enable_web_search requires ollama_api_key to be set. "
                "Configure DOCMIND_OLLAMA_API_KEY."
            )

        if self.security.allow_remote_endpoints:
            return

        allowed_hosts = parse_endpoint_allowlist_hosts(self.security.endpoint_allowlist)
        if not endpoint_url_allowed(
            "https://ollama.com",
            allowed_hosts=allowed_hosts,
        ):
            raise ValueError(
                "ollama_enable_web_search requires https://ollama.com in "
                "security.endpoint_allowlist or allow_remote_endpoints=True."
            )

    @computed_field
    @property
    def app_version(self) -> str:
        """Return the release-owned package version.

        Returns:
            str: Installed distribution or source-project version.
        """
        return get_version()


_SPACY_ENV_BRIDGE: dict[str, str] = {
    "SPACY_ENABLED": "DOCMIND_SPACY__ENABLED",
    "SPACY_MODEL": "DOCMIND_SPACY__MODEL",
    "SPACY_DEVICE": "DOCMIND_SPACY__DEVICE",
    "SPACY_GPU_ID": "DOCMIND_SPACY__GPU_ID",
    "SPACY_DISABLE_PIPES": "DOCMIND_SPACY__DISABLE_PIPES",
    "SPACY_BATCH_SIZE": "DOCMIND_SPACY__BATCH_SIZE",
    "SPACY_N_PROCESS": "DOCMIND_SPACY__N_PROCESS",
    "SPACY_MAX_CHARACTERS": "DOCMIND_SPACY__MAX_CHARACTERS",
}


def _apply_spacy_env_bridge() -> None:
    """Bridge flat `SPACY_*` env vars into `DOCMIND_SPACY__*`.

    This keeps the app aligned with DocMind's nested settings while supporting
    the required "SPACY_*" knobs for operators.
    """
    for src, dest in _SPACY_ENV_BRIDGE.items():
        if dest not in os.environ and src in os.environ:
            os.environ[dest] = os.environ[src]


# Global settings instance - primary interface for the application
_apply_spacy_env_bridge()
settings = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

_DOTENV_BOOTSTRAPPED = False


def reset_bootstrap_state() -> None:
    """Reset dotenv/bootstrap global state (tests only)."""
    global _DOTENV_BOOTSTRAPPED, _DOTENV_PRIORITY_MODE
    _DOTENV_BOOTSTRAPPED = False
    _DOTENV_PRIORITY_MODE = None


def bootstrap_settings(
    *, env_file: Path | None = None, force: bool = False
) -> Path | None:
    """Optionally load dotenv into the process-global settings singleton.

    This is an explicit startup step (no import-time dotenv IO). It is safe to call
    multiple times; subsequent calls are no-ops unless `force=True`.

    Args:
        env_file: Optional dotenv path override.
        force: When True, reload even if already bootstrapped.

    Returns:
        The dotenv path used when loading occurred, otherwise None.
    """
    global _DOTENV_BOOTSTRAPPED
    if _DOTENV_BOOTSTRAPPED and not force:
        return None

    dotenv_path = env_file if env_file is not None else resolve_dotenv_path()
    try:
        exists = dotenv_path.is_file()
    except OSError:
        exists = False

    if not exists:
        _DOTENV_BOOTSTRAPPED = True
        return None

    global _DOTENV_PRIORITY_MODE
    opts = _read_bootstrap_options(dotenv_path)
    _DOTENV_PRIORITY_MODE = opts.dotenv_priority

    # Optional: remove selected global env vars so third-party libs won't pick them up.
    _apply_env_mask(opts.env_mask_keys)

    _apply_spacy_env_bridge()
    settings.__init__(_env_file=dotenv_path)  # type: ignore[arg-type]

    # Optional: overlay allowlisted env vars from validated settings values.
    _apply_env_overlay(opts.env_overlays)
    _DOTENV_BOOTSTRAPPED = True
    return dotenv_path


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
    "DocumentParsingConfig",
    "EmbeddingConfig",
    "GraphRAGConfig",
    "ImageEncryptionConfig",
    "LLMRequestConfig",
    "OcrConfig",
    "PdfBackendConfig",
    "ProcessingConfig",
    "RetrievalConfig",
    "SpacyNlpSettings",
    "TelemetryConfig",
    "UIConfig",
    "bootstrap_settings",
    "settings",
]
