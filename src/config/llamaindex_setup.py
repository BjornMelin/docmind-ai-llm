"""LlamaIndex framework configuration setup.

Configures LlamaIndex Settings with environment variables for LLM,
embedding, and retrieval configuration following ADR-002, ADR-004,
ADR-010, and ADR-024 specifications.

Key Features:
- BGE-M3 unified dense/sparse embeddings with FP16 optimization (ADR-002)
- Qwen3-4B-Instruct-2507-FP8 with 128K context window (ADR-004)
- vLLM FP8 KV cache optimization for RTX 4090 (ADR-010)

Usage:
    >>> from src.config.llamaindex_setup import setup_llamaindex
    >>> setup_llamaindex()  # Call once at application startup
"""

import logging
import os
from pathlib import Path

import torch
from llama_index.core import Settings

# Removed SentenceSplitter import - ADR-009 requires direct Unstructured.io
# chunk_by_title
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from .app_settings import app_settings

logger = logging.getLogger(__name__)


def setup_llamaindex() -> None:
    """Configure LlamaIndex Settings with environment variables.

    Handles LLM, embedding, and document processing configuration
    through LlamaIndex's native Settings system.
    """
    # Configure LLM
    try:
        Settings.llm = Ollama(
            model=os.getenv("DOCMIND_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507-FP8"),
            base_url=app_settings.ollama_base_url,
            temperature=float(os.getenv("DOCMIND_TEMPERATURE", "0.1")),
            top_p=float(os.getenv("DOCMIND_TOP_P", "0.8")),
            top_k=int(os.getenv("DOCMIND_TOP_K", "40")),
            request_timeout=app_settings.request_timeout_seconds,
        )
        logger.info("LLM configured: %s", Settings.llm.model)
    except Exception as e:
        logger.warning("Could not configure LLM: %s", e)
        Settings.llm = None

    # Configure embeddings with BGE-M3 optimizations (ADR-002)
    try:
        embedding_model = app_settings.bge_m3_model_name
        use_gpu = app_settings.enable_gpu_acceleration

        # BGE-M3 specific configuration from ADR-002
        torch_dtype = (
            torch.float16 if (use_gpu and torch.cuda.is_available()) else torch.float32
        )

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            device="cuda" if use_gpu else "cpu",
            cache_folder=str(Path("./embeddings_cache").resolve()),
            max_length=app_settings.bge_m3_max_length,
            embed_batch_size=app_settings.bge_m3_batch_size_gpu
            if use_gpu
            else app_settings.bge_m3_batch_size_cpu,
            trust_remote_code=True,  # Required for BGE-M3 (ADR-002)
            torch_dtype=torch_dtype,  # FP16 optimization for GPU (ADR-002)
        )
        logger.info(
            "Embedding model configured: %s (device=%s, dtype=%s)",
            embedding_model,
            "cuda" if use_gpu else "cpu",
            torch_dtype,
        )
    except Exception as e:
        logger.warning("Could not configure embeddings: %s", e)
        Settings.embed_model = None

    # Document processing removed - ADR-009 requires direct Unstructured.io
    # chunk_by_title() semantic intelligence instead of SentenceSplitter
    # New implementation will use ResilientDocumentProcessor with direct
    # partition() and chunk_by_title() calls per specification

    # Configure context window and performance settings (ADR-004, ADR-010)
    try:
        context_window = int(os.getenv("DOCMIND_CONTEXT_WINDOW_SIZE", "131072"))
        max_tokens = int(os.getenv("DOCMIND_MAX_TOKENS", "2048"))

        # Core context settings
        Settings.context_window = context_window
        Settings.num_output = max_tokens

        # Set critical vLLM performance settings if not already set (ADR-010)
        vllm_defaults = {
            "VLLM_ATTENTION_BACKEND": app_settings.vllm_attention_backend,
            "VLLM_KV_CACHE_DTYPE": "fp8_e5m2",
            "VLLM_GPU_MEMORY_UTILIZATION": str(
                app_settings.vllm_gpu_memory_utilization
            ),
        }
        for key, value in vllm_defaults.items():
            if key not in os.environ:
                os.environ[key] = value

        logger.info(
            "Context configured: %d window, %d max tokens", context_window, max_tokens
        )
    except Exception as e:
        logger.warning("Could not set context configuration: %s", e)
