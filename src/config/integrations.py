"""LlamaIndex integration setup.

This module configures LlamaIndex global ``Settings`` from the unified
configuration (SPEC-001). It provides a single definitive binding point for
the active LLM backend via the factory.

Key behaviors:
- No network at import time
- Respect existing Settings unless forced
- Hardware-aware LlamaCPP params
"""

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from llama_index.core import Settings
from loguru import logger

from src.config.embedding_defaults import (
    BGE_M3_EMBEDDING_DIMENSION,
    DEFAULT_BGE_M3_MODEL_ID,
)
from src.config.llm_factory import build_llm
from src.telemetry.opentelemetry import setup_metrics, setup_tracing

from .settings import DocMindSettings, settings

if TYPE_CHECKING:  # pragma: no cover
    from llama_index.core.base.embeddings.base import BaseEmbedding
else:
    BaseEmbedding = Any


# Text embeddings default wrapper
# NOTE: Avoid heavy imports at module import-time. The embedding constructor
# symbol is bound lazily inside setup_llamaindex(). Callers may patch the module
# attribute HuggingFaceEmbedding; we preserve a stable module-level name for this.
HuggingFaceEmbedding: type[BaseEmbedding] | None = None  # type: ignore[assignment]
_HF_EMBED_LOCK = threading.Lock()

# Keep text embeddings defaulting to HuggingFaceEmbedding (BGE-M3) for
# consistency with tri-mode retrieval and VectorStoreIndex usage.

# Removed host-level checks; rely on centralized settings-side validation


def setup_llamaindex(*, force_llm: bool = False, force_embed: bool = False) -> None:
    """Configure global LlamaIndex settings based on DocMind configuration."""
    # Local-first defaults must be visible before any Hugging Face backed
    # constructor runs, otherwise cache hits can still perform network HEAD
    # probes before offline mode is enabled.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if _should_configure_llm(force_llm):
        _configure_llm()
    else:
        logger.info("LLM already configured; skipping override")

    if _should_configure_embeddings(force_embed):
        _configure_embeddings()
    else:
        logger.info("Embed model already configured; skipping override")

    _configure_context_settings()


def startup_init(cfg: "DocMindSettings" = settings) -> None:
    """Perform explicit startup initialization without import-time side effects.

    Ensures required directories exist, logs configuration highlights, and
    configures OpenTelemetry exporters when enabled.
    """
    try:
        # IO: ensure directories
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        cfg.cache.dir.mkdir(parents=True, exist_ok=True)
        cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Defensive: chat DB dir should not block startup
        try:
            cfg.chat.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            from src.utils.log_safety import build_pii_log_entry

            redaction = build_pii_log_entry(str(exc), key_id="startup.chat_db_dir")
            logger.warning(
                "Chat DB directory creation failed (non-blocking) for {} "
                "(error_type={} error={})",
                cfg.chat.sqlite_path.name,
                type(exc).__name__,
                redaction.redacted,
            )

        # Observability: log config highlights
        try:
            from src.utils.log_safety import safe_url_for_log

            base_url = cfg.backend_base_url_normalized
            safe_base_url = safe_url_for_log(str(base_url)) if base_url else ""
            logger.info(
                "Startup: backend={} base_url={} timeout={} hybrid={} fusion={}",
                cfg.llm_backend,
                safe_base_url,
                cfg.llm_request_timeout_seconds,
                cfg.retrieval.enable_server_hybrid,
                cfg.retrieval.fusion_mode,
            )
        except (
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:  # pragma: no cover - logging must not fail
            from src.utils.log_safety import build_pii_log_entry

            redaction = build_pii_log_entry(str(exc), key_id="startup.logging")
            logger.debug(
                "Startup logging failed (error_type={} error={})",
                type(exc).__name__,
                redaction.redacted,
            )
    except (
        OSError,
        AttributeError,
        TypeError,
        ValueError,
    ) as exc:  # pragma: no cover - defensive
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="startup.startup_init")
        logger.warning(
            "startup_init encountered error (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        return

    setup_tracing(cfg)
    setup_metrics(cfg)


def initialize_integrations(
    *, force_llm: bool = False, force_embed: bool = False
) -> None:
    """Initialize application startup services and LlamaIndex ``Settings``.

    Args:
        force_llm: Force-rebind ``Settings.llm``.
        force_embed: Force-rebind ``Settings.embed_model``.

    """
    startup_init(settings)
    setup_llamaindex(force_llm=force_llm, force_embed=force_embed)
    logger.info(
        "All integrations initialized successfully (force_llm={}, force_embed={})",
        force_llm,
        force_embed,
    )


# Convenience exports
__all__ = [
    "get_settings_embed_model",
    "initialize_integrations",
    "is_embedding_ready",
    "setup_llamaindex",
]


def _should_configure_llm(force_llm: bool) -> bool:
    return force_llm or getattr(Settings, "_llm", None) is None


def _configure_llm() -> None:
    try:
        settings._validate_endpoints_security()
    except ValueError as err:
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(err), key_id="integrations.llm_security")
        logger.warning(
            "LLM configuration blocked by security policy (error_type={} error={})",
            type(err).__name__,
            redaction.redacted,
        )
        raise

    try:
        Settings.llm = build_llm(settings)
        provider = settings.llm_backend
        model_name = settings.effective_model
        from src.utils.log_safety import safe_url_for_log

        base_url = getattr(settings, "backend_base_url_normalized", None)
        safe_base_url = safe_url_for_log(str(base_url)) if base_url else ""
        streaming = bool(getattr(settings, "llm_streaming_enabled", True))
        logger.info(
            "LLM configured via factory: provider={} model={} base_url={} streaming={}",
            provider,
            model_name,
            safe_base_url,
            streaming,
        )
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="integrations.configure_llm")
        logger.warning(
            "Could not configure LLM (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        raise


def get_settings_embed_model() -> BaseEmbedding | None:
    """Return the currently configured LlamaIndex embedding instance.

    Read LlamaIndex's backing slot so this check cannot lazily resolve OpenAI or
    a test-only ``MockEmbedding``.
    """
    try:
        return getattr(Settings, "_embed_model", None)
    except AttributeError as exc:  # pragma: no cover - defensive test doubles
        logger.debug(
            "Settings._embed_model unavailable (error_type={})",
            type(exc).__name__,
        )
        return None


def is_embedding_ready() -> bool:
    """Return whether a real LlamaIndex embedding model is configured."""
    from llama_index.core.base.embeddings.base import BaseEmbedding as LIBaseEmbedding
    from llama_index.core.embeddings import MockEmbedding

    embed_model = get_settings_embed_model()
    return isinstance(embed_model, LIBaseEmbedding) and not isinstance(
        embed_model, MockEmbedding
    )


def _should_configure_embeddings(force_embed: bool) -> bool:
    return force_embed or get_settings_embed_model() is None


def _configure_embeddings() -> None:
    try:
        emb_cfg = settings.get_embedding_config()
        model_id = emb_cfg.get("model_id", emb_cfg.get("model_name"))
        model_name = emb_cfg.get("model_name", DEFAULT_BGE_M3_MODEL_ID)
        device = emb_cfg.get("device", "cpu")
        dimension = int(emb_cfg.get("dimension", BGE_M3_EMBEDDING_DIMENSION))
        if (
            model_id == DEFAULT_BGE_M3_MODEL_ID
            and dimension != BGE_M3_EMBEDDING_DIMENSION
        ):
            raise ValueError(
                f"BAAI/bge-m3 requires embedding dimension {BGE_M3_EMBEDDING_DIMENSION}"
            )

        local_model_path = emb_cfg.get("local_model_path")
        if local_model_path is not None:
            local_path = Path(str(local_model_path)).expanduser()
            if not local_path.is_dir():
                raise FileNotFoundError(
                    f"Local embedding model directory not found: {local_path}"
                )
            model_name = str(local_path)

        global HuggingFaceEmbedding
        embedding_cls = HuggingFaceEmbedding
        if embedding_cls is None:
            with _HF_EMBED_LOCK:
                embedding_cls = HuggingFaceEmbedding
                if embedding_cls is None:
                    from llama_index.embeddings import (
                        huggingface as _hf,
                    )  # local import

                    HuggingFaceEmbedding = _hf.HuggingFaceEmbedding
                    embedding_cls = HuggingFaceEmbedding
        if embedding_cls is None:  # pragma: no cover - defensive
            raise RuntimeError("HuggingFaceEmbedding class unavailable")

        embedding_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "device": device,
            "max_length": int(emb_cfg.get("max_length", 8192)),
            "normalize": bool(emb_cfg.get("normalize_text", True)),
            "embed_batch_size": int(emb_cfg.get("batch_size_text", 4)),
            "trust_remote_code": emb_cfg.get("trust_remote_code", False),
            "local_files_only": bool(emb_cfg.get("local_files_only", True)),
        }
        cache_folder = emb_cfg.get("cache_folder")
        if cache_folder is not None:
            embedding_kwargs["cache_folder"] = str(cache_folder)
        model_revision = emb_cfg.get("model_revision")
        if local_model_path is None and model_revision is not None:
            embedding_kwargs["revision"] = str(model_revision)

        embed_model = cast(Any, embedding_cls)(**embedding_kwargs)
        Settings.embed_model = embed_model
        logger.info(
            "Embedding model configured: {} {} (device={})",
            type(embed_model).__name__,
            model_name,
            device,
        )
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="integrations.configure_embed")
        logger.warning(
            "Could not configure embeddings (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        raise


def _configure_context_settings() -> None:
    try:
        Settings.context_window = int(settings.llm_request.context_window)
        Settings.num_output = int(settings.llm_request.max_output_tokens)
        logger.info(
            "Context configured: {} window, {} max tokens",
            int(Settings.context_window),
            int(Settings.num_output),
        )
    except (AttributeError, ValueError) as exc:
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="integrations.context_cfg")
        logger.warning(
            "Could not set context configuration (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
