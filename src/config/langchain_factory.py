"""LangChain chat model factory for LangGraph workflows.

The core application uses LlamaIndex as the primary integration layer for
retrieval and offline-first backends. LangGraph (and langgraph-supervisor)
operate on LangChain `LanguageModelLike` runnables, so we provide a small
factory to build a compatible LangChain chat model from the same unified
settings.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config.settings import DocMindSettings, ensure_v1


def build_chat_model(cfg: DocMindSettings) -> ChatOpenAI:
    """Build a LangChain chat model for agent orchestration.

    Note: This uses the OpenAI-compatible API surface (``ChatOpenAI``) so it can
    target local servers (vLLM/LM Studio/llama.cpp server) via ``base_url``.
    For Ollama, we assume an OpenAI-compatible endpoint is available at ``/v1``.
    """
    model_name = cfg.model or cfg.vllm.model
    if not model_name:
        raise ValueError("No model name configured for LangChain model")
    timeout_s = float(
        getattr(cfg, "llm_request_timeout_seconds", cfg.ui.request_timeout_seconds)
    )

    base_url = cfg.backend_base_url_normalized
    if cfg.llm_backend == "ollama":
        base_url = ensure_v1(cfg.ollama_base_url)
    if not base_url:
        raise ValueError("No backend base URL configured for LangChain model")

    return ChatOpenAI(
        model=model_name,
        api_key=SecretStr(cfg.openai.api_key or "not-needed"),
        base_url=base_url,
        timeout=timeout_s,
        max_retries=int(cfg.agents.max_retries),
        temperature=float(cfg.vllm.temperature),
    )


__all__ = ["build_chat_model"]
