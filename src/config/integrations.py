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

import logging
import os
from contextlib import suppress

from llama_index.core import Settings

# Text embeddings should default to BGE-M3 (1024D) for consistency with
# tri-mode retrieval tooling and VectorStoreIndex usage. Use LlamaIndex's
# HuggingFaceEmbedding wrapper to keep a library-first, lightweight setup.
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config.llm_factory import build_llm

from .settings import settings

logger = logging.getLogger(__name__)


def _is_localhost(url: str | None) -> bool:
    """Return True if URL is localhost/127.0.0.1.

    Empty/None returns True (treated as local default).
    """
    if not url:
        return True
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        return host in {"localhost", "127.0.0.1"}
    except Exception:  # pragma: no cover - defensive
        return False


def setup_llamaindex(*, force_llm: bool = False, force_embed: bool = False) -> None:
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
    if Settings.llm is not None and not force_llm:
        logger.info("LLM already configured; skipping override")
    else:
        try:
            Settings.llm = build_llm(settings)
            # Observability: log provider + model + base_url once
            provider = settings.llm_backend
            model_name = settings.model or settings.vllm.model
            base_url: str | None
            if provider == "ollama":
                base_url = settings.ollama_base_url
            elif provider == "lmstudio":
                base_url = settings.lmstudio_base_url
            elif provider == "vllm":
                base_url = settings.vllm_base_url or settings.vllm.vllm_base_url
            elif provider == "llamacpp":
                base_url = settings.llamacpp_base_url or str(
                    settings.vllm.llamacpp_model_path
                )
            else:
                base_url = None

            # Enforce local-only endpoints unless allowlist override
            allow_remote = os.getenv("DOCMIND_ALLOW_REMOTE_ENDPOINTS", "").lower() in {
                "1",
                "true",
                "yes",
            }
            if (not allow_remote) and (not _is_localhost(base_url)):
                raise ValueError(
                    "Remote endpoint forbidden by default. Set "
                    "DOCMIND_ALLOW_REMOTE_ENDPOINTS=true to override."
                )
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
            emb_cfg = settings.get_embedding_config()
            model_name = emb_cfg.get("model_name", "BAAI/bge-m3")
            device = emb_cfg.get("device", "cpu")
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
            )
            logger.info(
                "Embedding model configured: HuggingFaceEmbedding %s (device=%s)",
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
        "--trust-remote-code",
    ]

    if settings.vllm.enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")

    return cmd


def initialize_integrations(
    *, force_llm: bool = False, force_embed: bool = False
) -> None:
    """Initialize both vLLM environment and LlamaIndex ``Settings``.

    Args:
        force_llm: Force-rebind ``Settings.llm``.
        force_embed: Force-rebind ``Settings.embed_model``.

    Calls :func:`setup_vllm_env` and :func:`setup_llamaindex` in order.
    """
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
    from src.models.embeddings import ImageEmbedder, TextEmbedder, UnifiedEmbedder

    emb_cfg = settings.get_embedding_config()
    device = emb_cfg.get("device", "cpu")
    text = TextEmbedder(device=device)
    image = ImageEmbedder(device=device)
    return UnifiedEmbedder(text=text, image=image, strict_image_types=True)


def get_clip_like_image_embedder():  # pragma: no cover - convenience adapter
    """Adapter that exposes `get_image_embedding(img)` using UnifiedEmbedder.

    This makes it drop-in compatible with existing multimodal helpers that
    expect a `clip`-like object while avoiding heavy third-party imports.
    """
    import numpy as _np

    u = get_unified_embedder()

    class _ClipLike:
        def get_image_embedding(self, image: object) -> _np.ndarray:
            arr = u.image.encode_image([image])
            if arr.shape[0] == 0:
                dim = int(u.image._dim or 768)  # best-effort dimension
                return _np.zeros(dim, dtype=_np.float32)
            return arr[0]

    return _ClipLike()
