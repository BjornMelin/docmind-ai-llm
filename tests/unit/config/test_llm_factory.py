"""Tests for LLM factory configuration and building functionality."""

from unittest.mock import MagicMock, patch

import pytest

from src.config.llm_factory import build_llm
from src.config.settings import DocMindSettings


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["ollama", "vllm", "lmstudio"])
def test_build_llm_openai_like_and_ollama(backend):
    """Test LLM building for OpenAI-like backends (vllm, lmstudio) and Ollama."""
    cfg = DocMindSettings()
    cfg.llm_backend = backend
    cfg.vllm.model = "qwen2.5-7b-instruct"
    cfg.vllm.context_window = 8192
    cfg.llm_request_timeout_seconds = 123
    cfg.openai_like_api_key = "abc"
    cfg.openai_like_is_chat_model = True
    cfg.openai_like_is_function_calling_model = False

    if backend == "ollama":
        with patch("llama_index.llms.ollama.Ollama", autospec=True) as p:
            inst = MagicMock(name="OllamaInstance")
            p.return_value = inst
            out = build_llm(cfg)
            assert out is inst
            _, kwargs = p.call_args
            assert kwargs["base_url"] == cfg.ollama_base_url
            assert kwargs["model"] == cfg.vllm.model
            assert float(kwargs["request_timeout"]) == float(
                cfg.llm_request_timeout_seconds
            )

    elif backend == "vllm":
        with patch("llama_index.llms.openai_like.OpenAILike", autospec=True) as p:
            inst = MagicMock(name="OpenAILikeVLLM")
            p.return_value = inst
            out = build_llm(cfg)
            assert out is inst
            _, kwargs = p.call_args
            assert kwargs["api_base"] == cfg.vllm.vllm_base_url  # no /v1 by default
            assert kwargs["model"] == cfg.vllm.model
            assert kwargs["api_key"] == cfg.openai_like_api_key
            assert kwargs["is_chat_model"] is True
            assert kwargs["is_function_calling_model"] is False
            assert kwargs["context_window"] == cfg.vllm.context_window
            assert float(kwargs["timeout"]) == float(cfg.llm_request_timeout_seconds)

    elif backend == "lmstudio":
        with patch("llama_index.llms.openai_like.OpenAILike", autospec=True) as p:
            inst = MagicMock(name="OpenAILikeLMStudio")
            p.return_value = inst
            out = build_llm(cfg)
            assert out is inst
            _, kwargs = p.call_args
            assert kwargs["api_base"] == cfg.lmstudio_base_url
            assert kwargs["is_chat_model"] is True
            assert kwargs["is_function_calling_model"] is False
            assert kwargs["context_window"] == cfg.vllm.context_window


@pytest.mark.unit
@pytest.mark.parametrize("gpu", [True, False])
def test_build_llm_llamacpp_offload(gpu):
    """Test LlamaCPP LLM building with GPU acceleration configuration."""
    cfg = DocMindSettings()
    cfg.llm_backend = "llamacpp"
    cfg.enable_gpu_acceleration = gpu

    with patch("llama_index.llms.llama_cpp.LlamaCPP", autospec=True) as p:
        inst = MagicMock(name="LlamaCPP")
        p.return_value = inst
        out = build_llm(cfg)
        assert out is inst
        _, kwargs = p.call_args
        assert kwargs["model_path"] == str(cfg.vllm.llamacpp_model_path)
        assert kwargs["context_window"] == cfg.vllm.context_window
        assert kwargs["model_kwargs"]["n_gpu_layers"] == (-1 if gpu else 0)


def test_vllm_top_level_overrides_and_api_base_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level model/context take priority; api_base uses top-level vllm_base_url."""

    # Stub OpenAILike to capture constructor args
    class _OpenAILike:
        def __init__(
            self,
            *,
            model: str,
            api_base: str,
            context_window: int | None = None,
            timeout: float | int | None = None,
            **_: object,
        ) -> None:
            self.model = model
            self.api_base = api_base
            self.context_window = context_window
            self.timeout = timeout

    import types as _types

    openai_like_mod = _types.ModuleType("llama_index.llms.openai_like")
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
    """llama.cpp path uses n_gpu_layers based on acceleration setting."""

    class _LlamaCPP:
        def __init__(
            self,
            *,
            model_path: str,
            context_window: int | None = None,
            model_kwargs: dict[str, object] | None = None,
            **_: object,
        ) -> None:
            self.model_path = model_path
            self.context_window = context_window
            self.model_kwargs = model_kwargs or {}

    import types as _types

    llama_cpp_mod = _types.ModuleType("llama_index.llms.llama_cpp")
    llama_cpp_mod.LlamaCPP = _LlamaCPP  # type: ignore[attr-defined]
    monkeypatch.setitem(
        __import__("sys").modules, "llama_index.llms.llama_cpp", llama_cpp_mod
    )

    cfg1 = DocMindSettings(
        llm_backend="llamacpp",
        model="local-gguf-path",
        context_window=2048,
        enable_gpu_acceleration=True,
    )
    llm1 = build_llm(cfg1)
    assert getattr(llm1, "context_window", None) == 2048
    assert getattr(llm1, "model_kwargs", {}).get("n_gpu_layers") == -1

    cfg2 = DocMindSettings(
        llm_backend="llamacpp",
        model="local-gguf-path",
        context_window=1024,
        enable_gpu_acceleration=False,
    )
    llm2 = build_llm(cfg2)
    assert getattr(llm2, "context_window", None) == 1024
    assert getattr(llm2, "model_kwargs", {}).get("n_gpu_layers") == 0


def test_invalid_context_window_raises_value_error() -> None:
    """Invalid context_window raises ValueError via validation."""
    with pytest.raises(
        ValueError, match=r"(must be > 0|greater than or equal to 1024)"
    ):
        DocMindSettings(context_window=0)  # type: ignore[call-arg]
    with pytest.raises(
        ValueError, match=r"(must be > 0|greater than or equal to 1024)"
    ):
        DocMindSettings(context_window=-1024)  # type: ignore[call-arg]
