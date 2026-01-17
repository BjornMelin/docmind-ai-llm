"""LangChain chat model factory for LangGraph workflows.

The core application uses LlamaIndex as the primary integration layer for
retrieval and offline-first backends. LangGraph agent graphs operate on LangChain
`LanguageModelLike` runnables, so we provide a small
factory to build a compatible LangChain chat model from the same unified
settings.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config.settings import DocMindSettings
from src.config.settings_utils import ensure_v1


def build_chat_model(cfg: DocMindSettings) -> ChatOpenAI:
    """Build a LangChain chat model for agent orchestration.

    Note: This uses the OpenAI-compatible API surface (``ChatOpenAI``) so it can
    target local servers (vLLM/LM Studio/llama.cpp server) via ``base_url``.
    For Ollama, we assume an OpenAI-compatible endpoint is available at ``/v1``.

    Args:
        cfg: Loaded application settings.

    Returns:
        A configured `ChatOpenAI` runnable.

    Raises:
        ValueError: If no model name or base URL is configured.
    """
    model_name = cfg.model or cfg.vllm.model
    if not model_name:
        raise ValueError("No model name configured for LangChain model")
    timeout_s = float(
        getattr(cfg, "llm_request_timeout_seconds", cfg.ui.request_timeout_seconds)
    )
    if cfg.agents.enable_deadline_propagation:
        timeout_s = min(timeout_s, float(cfg.agents.decision_timeout))

    base_url = cfg.backend_base_url_normalized
    if cfg.llm_backend == "ollama":
        base_url = ensure_v1(cfg.ollama_base_url)
    if not base_url:
        raise ValueError("No backend base URL configured for LangChain model")

    use_responses = (
        cfg.llm_backend == "openai_compatible" and cfg.openai.api_mode == "responses"
    )
    output_version = "responses/v1" if use_responses else None

    return ChatOpenAI(
        model=model_name,
        api_key=cfg.openai.api_key or SecretStr("not-needed"),
        base_url=base_url,
        timeout=timeout_s,
        max_retries=int(cfg.agents.max_retries),
        temperature=float(cfg.vllm.temperature),
        default_headers=cfg.openai.default_headers,
        use_responses_api=use_responses,
        output_version=output_version,
    )


__all__ = ["build_chat_model"]
