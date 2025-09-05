"""Extended unit tests for LLM factory behavior.

Covers top-level overrides, base URL precedence, URL validation, and
llama.cpp local library parameters using safe stubs to avoid heavy deps.
"""

from __future__ import annotations

from types import ModuleType

import pytest

from src.config.llm_factory import build_llm
from src.config.settings import DocMindSettings


def test_vllm_top_level_overrides_and_api_base_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level model/context take priority; api_base uses top-level vllm_base_url."""
    # Stub OpenAILike to capture constructor args
    captured: dict = {}

    class _OpenAILike:
        def __init__(self, *, model, api_base, context_window=None, timeout=None, **_):  # type: ignore[no-untyped-def]
            captured.update(
                {
                    "model": model,
                    "api_base": api_base,
                    "context_window": context_window,
                    "timeout": timeout,
                }
            )
            # store attributes for potential downstream assertions
            self.model = model
            self.api_base = api_base
            self.context_window = context_window
            self.timeout = timeout

    openai_like_mod = ModuleType("llama_index.llms.openai_like")
    openai_like_mod.OpenAILike = _OpenAILike  # type: ignore[attr-defined]
    monkeypatch.setitem(
        __import__("sys").modules, "llama_index.llms.openai_like", openai_like_mod
    )

    cfg = DocMindSettings(
        llm_backend="vllm",
        model="Override-Model",
        context_window=4096,
        vllm_base_url="http://localhost:8000",
        llm_request_timeout_seconds=42,
    )

    llm = build_llm(cfg)
    assert getattr(llm, "api_base", None) == "http://localhost:8000"
    assert getattr(llm, "model", None) == "Override-Model"
    assert getattr(llm, "context_window", None) == 4096
    assert getattr(llm, "timeout", None) == 42.0


def test_lmstudio_url_must_end_with_v1() -> None:
    """LM Studio base URL must end with /v1; validation should raise."""
    with pytest.raises(ValueError, match="LM Studio base URL must end with /v1"):
        DocMindSettings(
            llm_backend="lmstudio", lmstudio_base_url="http://localhost:1234"
        )


def test_llamacpp_local_uses_gpu_layers_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """llama.cpp path uses n_gpu_layers.

    Based on enable_gpu_acceleration and context window.
    """
    # Stub LlamaCPP to capture params
    captured_local: dict = {}

    class _LlamaCPP:
        def __init__(self, *, model_path, context_window=None, model_kwargs=None, **_):  # type: ignore[no-untyped-def]
            captured_local.update(
                {
                    "model_path": model_path,
                    "context_window": context_window,
                    "model_kwargs": model_kwargs or {},
                }
            )
            self.model_path = model_path
            self.context_window = context_window
            self.model_kwargs = model_kwargs or {}

    llama_cpp_mod = ModuleType("llama_index.llms.llama_cpp")
    llama_cpp_mod.LlamaCPP = _LlamaCPP  # type: ignore[attr-defined]
    monkeypatch.setitem(
        __import__("sys").modules, "llama_index.llms.llama_cpp", llama_cpp_mod
    )

    # Case 1: GPU enabled → n_gpu_layers = -1
    cfg1 = DocMindSettings(
        llm_backend="llamacpp",
        model="local-gguf-path",
        context_window=2048,
        enable_gpu_acceleration=True,
        # ensure local path mode by leaving llamacpp_base_url unset
    )
    _ = build_llm(cfg1)
    assert captured_local.get("context_window") == 2048
    assert captured_local.get("model_kwargs", {}).get("n_gpu_layers") == -1

    # Case 2: CPU only → n_gpu_layers = 0
    cfg2 = DocMindSettings(
        llm_backend="llamacpp",
        model="local-gguf-path",
        context_window=1024,
        enable_gpu_acceleration=False,
    )
    _ = build_llm(cfg2)
    assert captured_local.get("context_window") == 1024
    assert captured_local.get("model_kwargs", {}).get("n_gpu_layers") == 0
