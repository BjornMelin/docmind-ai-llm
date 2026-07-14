"""LLM factory using official LlamaIndex integrations.

Builds an LLM instance based on the configured backend. We intentionally
avoid performing any network I/O at import time and prefer library-first
adapters provided by LlamaIndex.

Backends:
- ``ollama``   -> :class:`llama_index.llms.ollama.Ollama`
- ``openai_compatible`` -> :class:`llama_index.llms.openai_like.OpenAILike` (or
  :class:`llama_index.llms.openai.OpenAIResponses` when enabled)
- ``vllm``     -> :class:`llama_index.llms.openai_like.OpenAILike`
- ``lmstudio`` -> :class:`llama_index.llms.openai_like.OpenAILike`
- ``llamacpp`` -> :class:`llama_index.llms.openai_like.OpenAILike`

Notes:
- All OpenAI-compatible servers (LM Studio, vLLM OpenAI-compatible, llama.cpp server)
  are normalized to include a single "/v1" by default. Disable normalization via
  ``settings.openai.require_v1`` only for endpoints rooted at "/" (e.g., LiteLLM Proxy).
"""

from __future__ import annotations

from typing import Any

from src.config.ollama_client import resolve_ollama_auth
from src.config.settings import DocMindSettings


def _resolve_api_key(settings: DocMindSettings) -> str:
    """Resolve the OpenAI-compatible API key from settings."""
    if settings.openai.api_key is not None:
        return settings.openai.api_key.get_secret_value()
    return "not-needed"


def build_llm(settings: DocMindSettings) -> Any:
    """Construct the appropriate LLM from settings.

    Args:
        settings: Application settings instance.

    Returns:
        A LlamaIndex LLM instance for the selected backend.

    Raises:
        ValueError: If ``settings.llm_backend`` is unsupported.
    """
    backend = settings.llm_backend
    supported_backends = {
        "ollama",
        "openai_compatible",
        "vllm",
        "lmstudio",
        "llamacpp",
    }
    if backend not in supported_backends:
        raise ValueError(f"Unsupported llm_backend: {backend}")

    model_name = settings.effective_model
    context_window = settings.effective_context_window
    request = settings.llm_request
    max_retries = int(settings.agents.max_retries)
    timeout_s = float(settings.llm_request_timeout_seconds)
    timeout_s = min(timeout_s, float(settings.agents.decision_timeout))

    llm: Any

    if backend == "ollama":
        from llama_index.llms.ollama import Ollama  # type: ignore

        _, ollama_headers = resolve_ollama_auth(settings)
        llm = Ollama(
            base_url=str(settings.ollama_base_url).rstrip("/"),
            model=model_name,
            request_timeout=timeout_s,
            context_window=context_window,
            temperature=float(request.temperature),
            additional_kwargs={"num_predict": int(request.max_output_tokens)},
            headers=ollama_headers or None,
        )
    elif backend == "openai_compatible":
        api_base = settings.backend_base_url_normalized
        if not api_base:
            raise ValueError("No OpenAI-compatible base URL configured")

        api_key = _resolve_api_key(settings)
        if settings.openai.api_mode == "responses":
            from llama_index.llms.openai import OpenAIResponses  # type: ignore

            # OpenAIResponses supports these parameters in llama-index-llms-openai.
            llm = OpenAIResponses(
                model=model_name,
                api_base=api_base,
                api_key=api_key,
                default_headers=settings.openai.default_headers,
                timeout=timeout_s,
                max_retries=max_retries,
                context_window=context_window,
                temperature=float(request.temperature),
                max_output_tokens=int(request.max_output_tokens),
            )
        else:
            from llama_index.llms.openai_like import OpenAILike  # type: ignore

            llm = OpenAILike(
                model=model_name,
                api_base=api_base,
                api_key=api_key,
                is_chat_model=True,
                is_function_calling_model=False,
                context_window=context_window,
                timeout=timeout_s,
                max_retries=max_retries,
                default_headers=settings.openai.default_headers,
                temperature=float(request.temperature),
                max_tokens=int(request.max_output_tokens),
            )
    elif backend in {"vllm", "lmstudio", "llamacpp"}:
        from llama_index.llms.openai_like import OpenAILike  # type: ignore

        api_base = settings.backend_base_url_normalized
        if not api_base:
            raise ValueError(f"No OpenAI-compatible base URL configured for {backend}")

        llm = OpenAILike(
            model=model_name,
            api_base=api_base,
            api_key="not-needed",
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=context_window,
            timeout=timeout_s,
            max_retries=max_retries,
            default_headers=None,
            temperature=float(request.temperature),
            max_tokens=int(request.max_output_tokens),
        )
    else:
        raise ValueError(f"Unsupported llm_backend: {backend}")

    return llm


__all__ = ["build_llm"]
