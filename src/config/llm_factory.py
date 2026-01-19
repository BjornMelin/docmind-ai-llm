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
- ``llamacpp`` -> :class:`llama_index.llms.llama_cpp.LlamaCPP`

Notes:
- All OpenAI-compatible servers (LM Studio, vLLM OpenAI-compatible, llama.cpp server)
  are normalized to include a single "/v1" by default. Disable normalization via
  ``settings.openai.require_v1`` only for endpoints rooted at "/" (e.g., LiteLLM Proxy).
- LlamaCPP GPU offload must be passed through ``model_kwargs={"n_gpu_layers": ...}``.
"""

from __future__ import annotations

from typing import Any

from src.config.settings import DocMindSettings, OpenAIConfig

_DEFAULT_OPENAI_BASE_URL = OpenAIConfig().base_url


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
    model_name = settings.model or settings.vllm.model
    context_window = int(settings.context_window or settings.vllm.context_window)
    timeout_s = float(
        getattr(
            settings, "llm_request_timeout_seconds", settings.ui.request_timeout_seconds
        )
    )
    agents = getattr(settings, "agents", None)
    if getattr(agents, "enable_deadline_propagation", False):
        timeout_s = min(
            timeout_s, float(getattr(agents, "decision_timeout", timeout_s))
        )

    llm: Any

    if backend == "ollama":
        from llama_index.llms.ollama import Ollama  # type: ignore

        llm = Ollama(
            base_url=str(settings.ollama_base_url).rstrip("/"),
            model=model_name,
            request_timeout=timeout_s,
            context_window=context_window,
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
                context_window=context_window,
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
                default_headers=settings.openai.default_headers,
            )
    elif backend in {"vllm", "lmstudio"}:
        from llama_index.llms.openai_like import OpenAILike  # type: ignore

        llm = OpenAILike(
            model=model_name,
            api_base=settings.backend_base_url_normalized,
            api_key=_resolve_api_key(settings),
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=context_window,
            timeout=timeout_s,
            default_headers=settings.openai.default_headers,
        )
    elif backend == "llamacpp":
        openai_base_url = settings.openai.base_url
        has_custom_openai_base = bool(
            openai_base_url and openai_base_url != _DEFAULT_OPENAI_BASE_URL
        )
        if settings.llamacpp_base_url or has_custom_openai_base:
            from llama_index.llms.openai_like import OpenAILike  # type: ignore

            llm = OpenAILike(
                model=model_name,
                api_base=settings.backend_base_url_normalized,
                api_key=_resolve_api_key(settings),
                is_chat_model=True,
                is_function_calling_model=False,
                context_window=context_window,
                timeout=timeout_s,
                default_headers=settings.openai.default_headers,
            )
        else:
            from llama_index.llms.llama_cpp import LlamaCPP  # type: ignore

            llm = LlamaCPP(
                model_path=str(settings.vllm.llamacpp_model_path),
                context_window=context_window,
                model_kwargs={
                    "n_gpu_layers": -1 if settings.enable_gpu_acceleration else 0,
                },
            )
    else:
        raise ValueError(f"Unsupported llm_backend: {backend}")

    return llm


__all__ = ["build_llm"]
