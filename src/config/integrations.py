"""LlamaIndex and vLLM integration setup.

This module configures LlamaIndex global ``Settings`` and vLLM environment
variables using the unified configuration. It intentionally replaces complex
bespoke setup code with clear, KISS functions to ensure reliability and ease of
testing.
"""

import logging
import os

import torch
from llama_index.core import Settings

from src.retrieval.embeddings import BGEM3Embedding

from .settings import settings

logger = logging.getLogger(__name__)


def setup_llamaindex() -> None:
    """Configure LlamaIndex ``Settings`` with unified configuration.

    Sets up the global LLM and embedding model. Heavy dependencies are lazily
    imported and failures result in ``Settings.llm``/``Settings.embed_model``
    being set to ``None``.

    Raises:
        None: All errors are logged and converted to ``None`` assignments so the
        rest of the app can gracefully degrade.
    """
    # Configure LLM with unified settings (do not overwrite test/fixture values)
    try:
        if Settings.llm is not None:
            logger.info("LLM already configured; skipping override")
        else:
            # Lazy import to avoid heavy dependency at module import time
            from llama_index.llms.ollama import Ollama  # type: ignore

            model_config = settings.get_model_config()

            Settings.llm = Ollama(
                model=model_config["model_name"],
                base_url=model_config["base_url"],
                temperature=model_config["temperature"],
                request_timeout=120.0,
            )
            logger.info("LLM configured: %s", getattr(Settings.llm, "model", "unknown"))
    except (ImportError, RuntimeError, ValueError, OSError) as e:
        logger.warning("Could not configure LLM: %s", e, exc_info=True)
        Settings.llm = None

    # Configure BGE-M3 embeddings with unified settings
    # (do not overwrite test/fixture values)
    try:
        if Settings.embed_model is not None:
            logger.info("Embed model already configured; skipping override")
        else:
            embedding_config = settings.get_embedding_config()

            # Prefer FP16 when on CUDA; BGEM3 internally handles dtype
            use_fp16 = (
                embedding_config["device"] == "cuda" and torch.cuda.is_available()
            )

            Settings.embed_model = BGEM3Embedding(
                model_name=embedding_config["model_name"],
                device=embedding_config["device"],
                max_length=embedding_config["max_length"],
                batch_size=embedding_config["batch_size"],
                use_fp16=use_fp16,
            )
            logger.info(
                "Embedding model configured: %s (device=%s, fp16=%s)",
                embedding_config.get("model_name"),
                embedding_config.get("device"),
                use_fp16,
            )
    except (ImportError, RuntimeError, ValueError, OSError) as e:
        logger.warning("Could not configure embeddings: %s", e, exc_info=True)
        Settings.embed_model = None

    # Set context window and performance settings
    try:
        Settings.context_window = settings.vllm.context_window
        Settings.num_output = settings.vllm.max_tokens

        logger.info(
            "Context configured: %d window, %d max tokens",
            int(Settings.context_window),
            int(Settings.num_output),
        )
    except (AttributeError, ValueError) as e:
        logger.warning("Could not set context configuration: %s", e, exc_info=True)


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


def initialize_integrations() -> None:
    """Initialize both vLLM environment and LlamaIndex ``Settings``.

    Calls :func:`setup_vllm_env` and :func:`setup_llamaindex` in order.
    """
    setup_vllm_env()
    setup_llamaindex()
    logger.info("All integrations initialized successfully")


# Convenience exports
__all__ = [
    "get_vllm_server_command",
    "initialize_integrations",
    "setup_llamaindex",
    "setup_vllm_env",
]
