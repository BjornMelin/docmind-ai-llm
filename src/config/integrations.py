"""LlamaIndex and vLLM integration setup.

Simplified integration module that configures LlamaIndex Settings
with the unified configuration architecture. Replaces the complex
vllm_config.py and kv_cache.py modules with simple environment
variable setup.
"""

import logging
import os
from pathlib import Path

import torch
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from .settings import settings

logger = logging.getLogger(__name__)


def setup_llamaindex() -> None:
    """Configure LlamaIndex Settings with unified configuration.

    Sets up LLM and embedding models using the simplified configuration
    approach. All complex vLLM configuration is now handled through
    environment variables.
    """
    # Configure LLM with unified settings
    try:
        model_config = settings.get_model_config()

        Settings.llm = Ollama(
            model=model_config["model_name"],
            base_url=model_config["base_url"],
            temperature=model_config["temperature"],
            request_timeout=120.0,
        )
        logger.info("LLM configured: %s", Settings.llm.model)
    except (KeyError, ValueError, ConnectionError, ImportError) as e:
        logger.warning("Could not configure LLM: %s", e)
        Settings.llm = None

    # Configure BGE-M3 embeddings with unified settings
    try:
        embedding_config = settings.get_embedding_config()

        # Determine torch dtype for FP16 optimization (ADR-002)
        torch_dtype = (
            torch.float16
            if (embedding_config["device"] == "cuda" and torch.cuda.is_available())
            else torch.float32
        )

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embedding_config["model_name"],
            device=embedding_config["device"],
            cache_folder=str(Path("./embeddings_cache").resolve()),
            max_length=embedding_config["max_length"],
            embed_batch_size=embedding_config["batch_size"],
            trust_remote_code=embedding_config["trust_remote_code"],
        )
        logger.info(
            "Embedding model configured: %s (device=%s, dtype=%s)",
            embedding_config["model_name"],
            embedding_config["device"],
            torch_dtype,
        )
    except (KeyError, ValueError, OSError, ImportError, RuntimeError) as e:
        logger.warning("Could not configure embeddings: %s", e)
        Settings.embed_model = None

    # Set context window and performance settings
    try:
        Settings.context_window = settings.vllm.context_window
        Settings.num_output = settings.vllm.max_tokens

        logger.info(
            "Context configured: %d window, %d max tokens",
            Settings.context_window,
            Settings.num_output,
        )
    except (AttributeError, ValueError) as e:
        logger.warning("Could not set context configuration: %s", e)


def setup_vllm_env() -> None:
    """Set up vLLM environment variables.

    Replaces the complex VLLMConfig and KVCacheManager classes
    with simple environment variable setup. All vLLM optimization
    is now handled through environment variables.
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
    """Generate vLLM server command with FP8 optimization.

    Returns simple command for starting vLLM server with all
    optimization settings from environment variables.

    Returns:
        List of command arguments for vLLM server
    """
    cmd = [
        "vllm",
        "serve",
        settings.vllm.model,
        "--max-model-len",
        str(settings.vllm.context_window),
        "--kv-cache-dtype",
        settings.vllm.kv_cache_dtype,
        "--gpu-memory-utilization",
        str(settings.vllm.gpu_memory_utilization),
        "--max-num-seqs",
        str(settings.vllm.max_num_seqs),
        "--max-num-batched-tokens",
        str(settings.vllm.max_num_batched_tokens),
        "--trust-remote-code",
    ]

    if settings.vllm.enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")

    return cmd


def initialize_integrations() -> None:
    """Initialize all integrations.

    Single function to set up both LlamaIndex and vLLM
    integrations with the unified configuration.
    """
    setup_vllm_env()
    setup_llamaindex()
    logger.info("All integrations initialized successfully")


# Convenience exports
__all__ = [
    "setup_llamaindex",
    "setup_vllm_env",
    "get_vllm_server_command",
    "initialize_integrations",
]
