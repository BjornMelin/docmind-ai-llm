"""LangChain chat model factory for LangGraph workflows.

The core application uses LlamaIndex as the primary integration layer for
retrieval and offline-first backends. LangGraph agent graphs operate on
LangChain `LanguageModelLike` runnables, so we provide a small factory to build
a compatible LangChain chat model from the same unified settings.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config.ollama_client import resolve_ollama_auth
from src.config.settings import DocMindSettings
from src.config.settings_utils import ensure_v1


def build_chat_model(
    cfg: DocMindSettings,
    *,
    timeout_cap: float | None = None,
) -> ChatOpenAI:
    """Build a LangChain chat model for agent orchestration.

    Note: This uses the OpenAI-compatible API surface (``ChatOpenAI``) so it can
    target local servers (vLLM/LM Studio/llama.cpp server) via ``base_url``.
    For Ollama, we assume an OpenAI-compatible endpoint is available at ``/v1``.

    Args:
        cfg: Loaded application settings.
        timeout_cap: Optional caller-owned request bound. Supplying one disables
            provider-internal retries so retry/backoff cannot multiply that budget.

    Returns:
        A configured `ChatOpenAI` runnable.

    Raises:
        ValueError: If no model name or base URL is configured.
    """
    model_name = cfg.effective_model
    if not model_name:
        raise ValueError("No model name configured for LangChain model")
    timeout_s = float(cfg.llm_request_timeout_seconds)
    timeout_s = min(timeout_s, float(cfg.agents.decision_timeout))
    if timeout_cap is not None:
        if timeout_cap <= 0:
            raise ValueError("timeout_cap must be greater than zero")
        timeout_s = min(timeout_s, float(timeout_cap))
    max_retries = 0 if timeout_cap is not None else int(cfg.agents.max_retries)

    base_url = cfg.backend_base_url_normalized
    api_key: SecretStr | None = None
    default_headers: dict[str, str] | None = None
    if cfg.llm_backend == "openai_compatible":
        api_key = cfg.openai.api_key
        default_headers = cfg.openai.default_headers
    if cfg.llm_backend == "ollama":
        base_url = ensure_v1(cfg.ollama_base_url)
        ollama_api_key, ollama_headers = resolve_ollama_auth(cfg)
        api_key = SecretStr(ollama_api_key) if ollama_api_key else None
        default_headers = ollama_headers or None
    if not base_url:
        raise ValueError("No backend base URL configured for LangChain model")

    use_responses = (
        cfg.llm_backend == "openai_compatible" and cfg.openai.api_mode == "responses"
    )
    output_version = "responses/v1" if use_responses else None

    token_limit_kwargs: dict[str, Any]
    if cfg.llm_backend in {"ollama", "lmstudio", "llamacpp"}:
        token_limit_kwargs = {
            "extra_body": {"max_tokens": int(cfg.llm_request.max_output_tokens)}
        }
    else:
        token_limit_kwargs = {
            "max_completion_tokens": int(cfg.llm_request.max_output_tokens)
        }

    return ChatOpenAI(
        model=model_name,
        api_key=api_key or SecretStr("not-needed"),
        base_url=base_url,
        timeout=timeout_s,
        max_retries=max_retries,
        temperature=float(cfg.llm_request.temperature),
        default_headers=default_headers,
        use_responses_api=use_responses,
        output_version=output_version,
        **token_limit_kwargs,
    )


__all__ = ["build_chat_model"]
