"""LLM factory using official LlamaIndex integrations.

Builds an LLM instance based on the configured backend. We intentionally
avoid performing any network I/O at import time and prefer library-first
adapters provided by LlamaIndex.

Backends:
- ``ollama``   -> :class:`llama_index.llms.ollama.Ollama`
- ``vllm``     -> :class:`llama_index.llms.openai_like.OpenAILike`
- ``lmstudio`` -> :class:`llama_index.llms.openai_like.OpenAILike`
- ``llamacpp`` -> :class:`llama_index.llms.llama_cpp.LlamaCPP`

Notes:
- All OpenAI-compatible servers (LM Studio, vLLM OpenAI-compatible, llama.cpp server)
  must use base URLs that include a single "/v1". Normalization is applied centrally
  via settings.backend_base_url_normalized.
- LlamaCPP GPU offload must be passed through ``model_kwargs={"n_gpu_layers": ...}``.
"""

from __future__ import annotations

from typing import Any

from src.config.settings import DocMindSettings


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
    # Resolve model/context/timeout with top-level overrides when provided
    model_name = settings.model or settings.vllm.model
    context_window = int(settings.context_window or settings.vllm.context_window)
    timeout_s = float(
        getattr(
            settings, "llm_request_timeout_seconds", settings.ui.request_timeout_seconds
        )
    )

    if backend == "ollama":
        from llama_index.llms.ollama import Ollama  # type: ignore

        return Ollama(
            base_url=settings.ollama_base_url,
            model=model_name,
            request_timeout=timeout_s,
            context_window=context_window,
        )

    if backend == "vllm":
        from llama_index.llms.openai_like import OpenAILike  # type: ignore

        # Always use the normalized backend base URL for OpenAI-like servers
        api_base = settings.backend_base_url_normalized
        return OpenAILike(
            model=model_name,
            api_base=api_base,
            api_key=(settings.openai.api_key or "not-needed"),
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=context_window,
            timeout=timeout_s,
        )

    if backend == "lmstudio":
        from llama_index.llms.openai_like import OpenAILike  # type: ignore

        return OpenAILike(
            model=model_name,
            api_base=settings.backend_base_url_normalized,
            api_key=(settings.openai.api_key or "not-needed"),
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=context_window,
            timeout=timeout_s,
        )

    if backend == "llamacpp":
        # Support both server (OpenAI-compatible) and local library
        if settings.llamacpp_base_url or settings.openai.base_url:
            from llama_index.llms.openai_like import OpenAILike  # type: ignore

            return OpenAILike(
                model=model_name,
                api_base=settings.backend_base_url_normalized,
                api_key=(settings.openai.api_key or "not-needed"),
                is_chat_model=True,
                is_function_calling_model=False,
                context_window=context_window,
                timeout=timeout_s,
            )

        from llama_index.llms.llama_cpp import LlamaCPP  # type: ignore

        return LlamaCPP(
            model_path=str(settings.vllm.llamacpp_model_path),
            context_window=context_window,
            model_kwargs={
                "n_gpu_layers": -1 if settings.enable_gpu_acceleration else 0,
            },
        )

    raise ValueError(f"Unsupported llm_backend: {backend}")


__all__ = ["build_llm"]
