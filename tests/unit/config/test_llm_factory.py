"""Tests for LLM factory configuration and building functionality."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from src.config.llm_factory import build_llm
from src.config.settings import DocMindSettings


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["ollama", "openai_compatible", "vllm", "lmstudio"])
def test_build_llm_openai_like_and_ollama(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test LLM building for OpenAI-like backends and Ollama."""
    openai_like_mod = ModuleType("llama_index.llms.openai_like")
    ollama_mod = ModuleType("llama_index.llms.ollama")

    class _CaptureOpenAILike:
        last_kwargs: dict[str, object] | None = None

        def __init__(self, **kwargs: object) -> None:
            type(self).last_kwargs = dict(kwargs)

    class _CaptureOllama:
        last_kwargs: dict[str, object] | None = None

        def __init__(self, **kwargs: object) -> None:
            type(self).last_kwargs = dict(kwargs)

    openai_like_mod.OpenAILike = _CaptureOpenAILike  # type: ignore[attr-defined]
    ollama_mod.Ollama = _CaptureOllama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_index.llms.openai_like", openai_like_mod)
    monkeypatch.setitem(sys.modules, "llama_index.llms.ollama", ollama_mod)

    cfg = DocMindSettings()
    cfg.llm_backend = backend
    cfg.vllm.model = "qwen2.5-7b-instruct"
    cfg.vllm.context_window = 8192
    cfg.llm_request_timeout_seconds = 123
    cfg.openai.api_key = "abc"

    if backend == "ollama":
        out = build_llm(cfg)
        kwargs = _CaptureOllama.last_kwargs or {}
        assert isinstance(out, _CaptureOllama)
        assert kwargs["base_url"] == str(cfg.ollama_base_url).rstrip("/")
        assert kwargs["model"] == cfg.vllm.model
        assert float(kwargs["request_timeout"]) == float(
            cfg.llm_request_timeout_seconds
        )

    elif backend in {"openai_compatible", "vllm"}:
        out = build_llm(cfg)
        kwargs = _CaptureOpenAILike.last_kwargs or {}
        assert isinstance(out, _CaptureOpenAILike)
        # Always normalized to include /v1
        assert str(kwargs["api_base"]).endswith("/v1")
        assert kwargs["model"] == cfg.vllm.model
        assert kwargs["api_key"] == cfg.openai.api_key.get_secret_value()
        assert kwargs["is_chat_model"] is True
        assert kwargs["is_function_calling_model"] is False
        assert kwargs["context_window"] == cfg.vllm.context_window
        assert float(kwargs["timeout"]) == float(cfg.llm_request_timeout_seconds)

    elif backend == "lmstudio":
        out = build_llm(cfg)
        kwargs = _CaptureOpenAILike.last_kwargs or {}
        assert isinstance(out, _CaptureOpenAILike)
        assert str(kwargs["api_base"]).endswith("/v1")
        assert kwargs["is_chat_model"] is True
        assert kwargs["is_function_calling_model"] is False
        assert kwargs["context_window"] == cfg.vllm.context_window


@pytest.mark.unit
def test_build_llm_openai_compatible_responses_uses_openai_responses() -> None:
    """OpenAI-compatible backend uses OpenAIResponses when api_mode=responses."""
    cfg = DocMindSettings()
    cfg.llm_backend = "openai_compatible"
    cfg.vllm.model = "qwen2.5-7b-instruct"
    cfg.vllm.context_window = 8192
    cfg.llm_request_timeout_seconds = 123
    cfg.openai.base_url = "http://localhost:8000"
    cfg.openai.api_key = "abc"
    cfg.openai.api_mode = "responses"
    cfg.openai.default_headers = {"X-Test": "1"}

    with patch("llama_index.llms.openai.OpenAIResponses", autospec=True) as p:
        inst = MagicMock(name="OpenAIResponses")
        p.return_value = inst
        out = build_llm(cfg)
        assert out is inst
        _, kwargs = p.call_args
        assert str(kwargs["api_base"]).endswith("/v1")
        assert kwargs["model"] == cfg.vllm.model
        assert kwargs["api_key"] == cfg.openai.api_key.get_secret_value()
        assert kwargs["default_headers"] == {"X-Test": "1"}
        assert kwargs["context_window"] == cfg.vllm.context_window
        assert float(kwargs["timeout"]) == float(cfg.llm_request_timeout_seconds)


@pytest.mark.unit
def test_build_llm_llamacpp_server_uses_openai_like(monkeypatch: pytest.MonkeyPatch):
    """llama.cpp backend uses the OpenAI-compatible server adapter."""
    openai_like_mod = ModuleType("llama_index.llms.openai_like")

    class _CaptureOpenAILike:
        last_kwargs: dict[str, object] | None = None

        def __init__(self, **kwargs: object) -> None:
            type(self).last_kwargs = dict(kwargs)

    openai_like_mod.OpenAILike = _CaptureOpenAILike  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_index.llms.openai_like", openai_like_mod)

    cfg = DocMindSettings()
    cfg.llm_backend = "llamacpp"
    cfg.llamacpp_base_url = "http://localhost:8080/v1"  # type: ignore[assignment]

    out = build_llm(cfg)
    kwargs = _CaptureOpenAILike.last_kwargs or {}

    assert isinstance(out, _CaptureOpenAILike)
    assert kwargs["api_base"] == "http://localhost:8080/v1"
    assert kwargs["is_chat_model"] is True
    assert kwargs["is_function_calling_model"] is False


@pytest.mark.unit
def test_build_llm_llamacpp_requires_server_url() -> None:
    """llama.cpp backend no longer constructs an in-process GGUF adapter."""
    cfg = DocMindSettings()
    cfg.llm_backend = "llamacpp"

    with pytest.raises(ValueError, match="No OpenAI-compatible base URL"):
        build_llm(cfg)


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
    assert str(getattr(llm, "api_base", None)).endswith("/v1")
    assert getattr(llm, "model", None) == "Override-Model"
    assert getattr(llm, "context_window", None) == 4096
    assert getattr(llm, "timeout", None) == 42.0


@pytest.mark.parametrize(
    ("input_url", "expected_url"),
    [
        ("http://localhost:1234", "http://localhost:1234/v1"),
        ("http://localhost:1234/v1", "http://localhost:1234/v1"),
        ("http://localhost:1234/v1/", "http://localhost:1234/v1"),
        ("http://localhost:1234///", "http://localhost:1234/v1"),
    ],
)
def test_lmstudio_url_normalized_to_include_v1(
    input_url: str,
    expected_url: str,
) -> None:
    """LM Studio base URL is normalized to include a single /v1 segment."""
    settings = DocMindSettings(llm_backend="lmstudio", lmstudio_base_url=input_url)
    assert str(settings.lmstudio_base_url).rstrip("/") == expected_url


def test_llamacpp_server_uses_context_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """llama.cpp server path uses OpenAI-compatible context and timeout."""

    class _OpenAILike:
        def __init__(
            self,
            *,
            api_base: str,
            context_window: int | None = None,
            timeout: float | int | None = None,
            **_: object,
        ) -> None:
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
        llm_backend="llamacpp",
        llamacpp_base_url="http://localhost:8080",
        context_window=2048,
        llm_request_timeout_seconds=17,
    )
    llm = build_llm(cfg)
    assert getattr(llm, "api_base", None) == "http://localhost:8080/v1"
    assert getattr(llm, "context_window", None) == 2048
    assert getattr(llm, "timeout", None) == 17.0


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
