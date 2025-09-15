"""LlamaIndex and vLLM integration setup.

This module configures LlamaIndex global ``Settings`` and vLLM environment
variables using the unified configuration (SPEC-001). It provides a single
definitive binding point for the active LLM backend via the factory.

Key behaviors:
- No network at import time
- Respect existing Settings unless forced
- Hardware-aware LlamaCPP params
- Optional structured-output flag for vLLM (SPEC-007 prep)
"""
# pylint: disable=ungrouped-imports

import logging
import os
import threading
from contextlib import suppress

from llama_index.core import Settings

from src.config.llm_factory import build_llm
from src.models.embeddings import ImageEmbedder, TextEmbedder, UnifiedEmbedder

from .settings import settings

# Text embeddings default wrapper
# NOTE: Avoid heavy imports at module import-time. The embedding constructor
# symbol is bound lazily inside setup_llamaindex(). Callers may patch the module
# attribute HuggingFaceEmbedding; we preserve a stable module-level name for this.
HuggingFaceEmbedding = None  # type: ignore  # pylint: disable=invalid-name
_HF_EMBED_LOCK = threading.Lock()

# Keep text embeddings defaulting to HuggingFaceEmbedding (BGE-M3) for
# consistency with tri-mode retrieval and VectorStoreIndex usage.

logger = logging.getLogger(__name__)


# Removed host-level checks; rely on centralized settings-side validation


def setup_llamaindex(*, force_llm: bool = False, force_embed: bool = False) -> None:  # pylint: disable=too-many-statements, too-many-branches
    """Configure LlamaIndex ``Settings`` with unified configuration.

    Args:
        force_llm: When True, rebind ``Settings.llm`` even if already set.
        force_embed: When True, rebind ``Settings.embed_model`` even if set.

    Notes:
        - Heavy libs are imported lazily. Errors are logged and converted to
          ``None`` assignments so the app can degrade gracefully.
        - Also sets ``Settings.context_window`` and ``Settings.num_output``.
        - Sets ``settings.guided_json_enabled`` when backend is vLLM.

    Flag Documentation:
        - force_llm: If True, forces reconfiguration of the global LLM even
          when one is already present. Useful in tests or when switching models
          at runtime via the Settings UI. Caveat: This affects the global
          ``Settings.llm``; in interactive or multi-user contexts, forcing a
          rebind can impact other users or ongoing sessions.
        - force_embed: If True, forces reconfiguration of the global embedding
          model even when one is already present. Similar caveats to ``force_llm``
          apply in multi-user scenarios.

        Best practice: Use force flags only when you are certain that overriding
        global configuration will not disrupt other users or in-flight requests.
    """
    # Configure LLM via factory
    has_llm = getattr(Settings, "_llm", None) is not None
    if has_llm and not force_llm:
        logger.info("LLM already configured; skipping override")
    else:
        try:
            Settings.llm = build_llm(settings)
            # Observability: log provider + model + base_url once
            provider = settings.llm_backend
            model_name = settings.model or settings.vllm.model
            base_url: str | None = getattr(
                settings, "backend_base_url_normalized", None
            )
            # Centralized endpoint security validation is enforced in settings
            logger.info(
                "LLM configured via factory: provider=%s model=%s base_url=%s",
                provider,
                model_name,
                base_url,
            )
            # Simple counters (log-based)
            logger.info("counter.provider_used: %s", provider)
            streaming = bool(getattr(settings, "llm_streaming_enabled", True))
            logger.info("counter.streaming_enabled: %s", streaming)
        except (ImportError, RuntimeError, ValueError, OSError) as e:
            logger.warning("Could not configure LLM: %s", e, exc_info=True)
            Settings.llm = None

    # Configure text embeddings (default to BGE-M3 1024D for global usage)
    try:
        if Settings.embed_model is not None and not force_embed:
            logger.info("Embed model already configured; skipping override")
        else:
            # Some environments may pass a lightweight settings namespace lacking
            # helper methods; default to BGE-M3 CPU in that case.
            if hasattr(settings, "get_embedding_config"):
                emb_cfg = settings.get_embedding_config()
            else:  # pragma: no cover - exercised in unit stubs
                emb_cfg = {"model_name": "BAAI/bge-m3", "device": "cpu"}
            model_name = emb_cfg.get("model_name", "BAAI/bge-m3")
            device = emb_cfg.get("device", "cpu")
            # Prefer HuggingFaceEmbedding for BGE-M3 (callers may monkeypatch
            # this constructor directly when needed). Import lazily when needed.
            global HuggingFaceEmbedding
            if HuggingFaceEmbedding is None:  # type: ignore
                # Double-checked locking to avoid repeated imports under concurrency
                with _HF_EMBED_LOCK:
                    if HuggingFaceEmbedding is None:  # type: ignore
                        from llama_index.embeddings import (
                            huggingface as _hf,  # local import
                        )

                        # Bind constructor once; callers may monkeypatch this symbol
                        HuggingFaceEmbedding = (
                            _hf.HuggingFaceEmbedding  # type: ignore[attr-defined]
                        )
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
                trust_remote_code=emb_cfg.get("trust_remote_code", False),
            )
            logger.info(
                "Embedding model configured: %s %s (device=%s)",
                type(Settings.embed_model).__name__,
                model_name,
                device,
            )
    except (ImportError, RuntimeError, ValueError, OSError) as e:
        logger.warning("Could not configure embeddings: %s", e, exc_info=True)
        Settings.embed_model = None

    # Set context window and performance settings (enforce global cap)
    try:
        Settings.context_window = min(
            int(settings.context_window or settings.vllm.context_window),
            int(settings.llm_context_window_max),
        )
        Settings.num_output = settings.vllm.max_tokens

        logger.info(
            "Context configured: %d window, %d max tokens",
            int(Settings.context_window),
            int(Settings.num_output),
        )
    except (AttributeError, ValueError) as e:
        logger.warning("Could not set context configuration: %s", e, exc_info=True)

    # Structured outputs capability flag (SPEC-007 prep)
    with suppress(Exception):  # pragma: no cover - defensive
        settings.guided_json_enabled = settings.llm_backend == "vllm"

    # Local-first defaults (offline and localhost enforcement)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def setup_vllm_env() -> None:
    """Set vLLM environment variables from unified settings.

    Uses environment variables to control vLLM optimization. Does not override
    existing variables already present in the process.
    """
    # Get vLLM environment variables from unified settings
    vllm_env = settings.get_vllm_env_vars()

    # Set environment variables for vLLM optimization
    for key, value in vllm_env.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.debug("Set %s=%s", key, value)

    logger.info("vLLM environment variables configured for FP8 optimization")


def get_vllm_server_command() -> list[str]:
    """Generate a vLLM server command with optimization flags.

    Returns:
        list[str]: Command line arguments for ``vllm serve`` using unified
        settings (context window, kv cache dtype, batching, etc.).
    """
    cmd = [
        "vllm",
        "serve",
        settings.vllm.model,
        "--max-model-len",
        str(int(settings.vllm.context_window)),
        "--kv-cache-dtype",
        settings.vllm.kv_cache_dtype,
        "--gpu-memory-utilization",
        str(float(settings.vllm.gpu_memory_utilization)),
        "--max-num-seqs",
        str(int(settings.vllm.max_num_seqs)),
        "--max-num-batched-tokens",
        str(int(settings.vllm.max_num_batched_tokens)),
    ]

    if settings.vllm.enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")

    return cmd


def startup_init(cfg: "settings.__class__" = settings) -> None:
    """Perform explicit startup initialization (no import-time side effects).

    - Ensure required directories exist
    - Bridge telemetry env only here if needed
    - Log resolved backend, normalized base URLs, timeouts, and hybrid mode
    """
    try:
        # IO: ensure directories
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
        if cfg.database.sqlite_db_path.parent != cfg.data_dir:
            cfg.database.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Telemetry env bridge (local JSONL sink still honored elsewhere)
        if not bool(getattr(cfg, "telemetry_enabled", True)):
            os.environ["DOCMIND_TELEMETRY_DISABLED"] = "true"

        # Observability: log config highlights
        try:
            from loguru import logger as _logger

            _logger.info(
                "Startup: backend=%s base_url=%s timeout=%s hybrid=%s fusion=%s",
                cfg.llm_backend,
                getattr(cfg, "backend_base_url_normalized", None),
                getattr(cfg, "llm_request_timeout_seconds", None),
                bool(getattr(cfg.retrieval, "enable_server_hybrid", False)),
                str(getattr(cfg.retrieval, "fusion_mode", "rrf")),
            )
        except Exception as exc:  # pragma: no cover - logging must not fail
            logger.debug("Startup logging failed: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive
        # Do not crash app on startup side-effects; callers may retry/log
        logger.warning("startup_init encountered error: %s", exc)


def initialize_integrations(
    *, force_llm: bool = False, force_embed: bool = False
) -> None:
    """Initialize both vLLM environment and LlamaIndex ``Settings``.

    Args:
        force_llm: Force-rebind ``Settings.llm``.
        force_embed: Force-rebind ``Settings.embed_model``.

    Calls :func:`setup_vllm_env` and :func:`setup_llamaindex` in order.
    """
    startup_init(settings)
    setup_vllm_env()
    setup_llamaindex(force_llm=force_llm, force_embed=force_embed)
    logger.info(
        "All integrations initialized successfully (force_llm=%s, force_embed=%s)",
        force_llm,
        force_embed,
    )


# Convenience exports
__all__ = [
    "get_vllm_server_command",
    "initialize_integrations",
    "setup_llamaindex",
    "setup_vllm_env",
]


# === Unified Embedder Factory (optional app wiring) ===
def get_unified_embedder():  # pragma: no cover - simple factory
    """Return a UnifiedEmbedder configured from settings.

    Uses the unified device selection and enables strict image type checks.
    This is a lightweight entry point for app code to opt into the
    library-first TextEmbedder (BGE-M3) and ImageEmbedder (OpenCLIP/SigLIP).
    """
    device = "cuda" if settings.enable_gpu_acceleration else "cpu"
    text = TextEmbedder(model_name=settings.embedding.model_name, device=device)
    image = ImageEmbedder(backbone="siglip_base", device=device)
    return UnifiedEmbedder(text=text, image=image, strict_image_types=True)


def get_image_embedder(
    backbone: str = "siglip_base",
) -> ImageEmbedder:  # pragma: no cover - simple factory
    """Return an ImageEmbedder with the requested backbone.

    Default backbone is SigLIP (preferred). Accepts "openclip_vitl14",
    "openclip_vith14", or "siglip_base".
    """
    device = "cuda" if settings.enable_gpu_acceleration else "cpu"
    return ImageEmbedder(backbone=backbone, device=device)


def get_clip_like_image_embedder():  # pragma: no cover - convenience adapter
    """Adapter that exposes `get_image_embedding(img)` using UnifiedEmbedder.

    This makes it drop-in compatible with existing multimodal helpers that
    expect a `clip`-like object while avoiding heavy third-party imports.
    """
    import numpy as _np

    u = get_unified_embedder()

    class _ClipLike:
        def get_image_embedding(self, image: object) -> _np.ndarray:
            """Return a single image embedding as a 1-D numpy array."""
            arr = u.image.encode_image([image])
            if arr.shape[0] == 0:
                # If embedder returned empty, fall back to a common 768-D shape
                return _np.zeros(768, dtype=_np.float32)
            return arr[0]

    return _ClipLike()
