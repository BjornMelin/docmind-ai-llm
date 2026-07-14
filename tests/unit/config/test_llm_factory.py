"""Tests for LLM factory configuration and building functionality."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from src.config.llm_factory import build_llm
from src.config.settings import DocMindSettings


class _CaptureOpenAILike:
    last_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = dict(kwargs)


class _CaptureOllama:
    last_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = dict(kwargs)


def _install_capture_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    openai_like_mod = ModuleType("llama_index.llms.openai_like")
    ollama_mod = ModuleType("llama_index.llms.ollama")
    openai_like_mod.OpenAILike = _CaptureOpenAILike  # type: ignore[attr-defined]
    ollama_mod.Ollama = _CaptureOllama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_index.llms.openai_like", openai_like_mod)
    monkeypatch.setitem(sys.modules, "llama_index.llms.ollama", ollama_mod)


@pytest.mark.unit
@pytest.mark.parametrize(
    "backend",
    ["ollama", "openai_compatible", "vllm", "lmstudio", "llamacpp"],
)
def test_build_llm_openai_like_and_ollama(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test LLM building for OpenAI-like backends and Ollama."""
    _install_capture_modules(monkeypatch)

    cfg = DocMindSettings()
    cfg.llm_backend = backend
    cfg.llm_request.model = "qwen2.5-7b-instruct"
    cfg.llm_request.context_window = 8192
    cfg.llm_request.max_output_tokens = 512
    cfg.llm_request.temperature = 0.2
    cfg.llm_request_timeout_seconds = 123
    cfg.openai.api_key = "abc"
    cfg.openai.default_headers = {"X-OpenAI": "must-not-leak"}
    if backend == "llamacpp":
        cfg.llamacpp_base_url = "http://localhost:8080/v1"

    if backend == "ollama":
        out = build_llm(cfg)
        kwargs = _CaptureOllama.last_kwargs or {}
        assert isinstance(out, _CaptureOllama)
        assert kwargs["base_url"] == str(cfg.ollama_base_url).rstrip("/")
        assert kwargs["model"] == cfg.effective_model
        assert float(kwargs["request_timeout"]) == float(
            cfg.llm_request_timeout_seconds
        )
        assert kwargs["headers"] is None
        assert kwargs["additional_kwargs"] == {"num_predict": 512}
        assert kwargs["temperature"] == pytest.approx(0.2)

    elif backend == "openai_compatible":
        out = build_llm(cfg)
        kwargs = _CaptureOpenAILike.last_kwargs or {}
        assert isinstance(out, _CaptureOpenAILike)
        # Always normalized to include /v1
        assert str(kwargs["api_base"]).endswith("/v1")
        assert kwargs["model"] == cfg.llm_request.model
        assert kwargs["api_key"] == cfg.openai.api_key.get_secret_value()
        assert kwargs["is_chat_model"] is True
        assert kwargs["is_function_calling_model"] is False
        assert kwargs["context_window"] == cfg.llm_request.context_window
        assert kwargs["max_tokens"] == cfg.llm_request.max_output_tokens
        assert kwargs["temperature"] == pytest.approx(cfg.llm_request.temperature)
        assert float(kwargs["timeout"]) == float(cfg.llm_request_timeout_seconds)
        assert kwargs["max_retries"] == cfg.agents.max_retries

    elif backend in {"vllm", "lmstudio", "llamacpp"}:
        out = build_llm(cfg)
        kwargs = _CaptureOpenAILike.last_kwargs or {}
        assert isinstance(out, _CaptureOpenAILike)
        assert str(kwargs["api_base"]).endswith("/v1")
        assert kwargs["is_chat_model"] is True
        assert kwargs["is_function_calling_model"] is False
        assert kwargs["context_window"] == cfg.llm_request.context_window
        assert kwargs["api_key"] == "not-needed"
        assert kwargs["default_headers"] is None


@pytest.mark.unit
def test_build_llm_uses_request_context_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider-neutral request owner supplies adapter context."""
    ollama_mod = ModuleType("llama_index.llms.ollama")

    class _CaptureOllama:
        last_kwargs: dict[str, object] | None = None

        def __init__(self, **kwargs: object) -> None:
            type(self).last_kwargs = dict(kwargs)

    ollama_mod.Ollama = _CaptureOllama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_index.llms.ollama", ollama_mod)

    cfg = DocMindSettings(
        llm_backend="ollama",
        llm_request={"context_window": 8_192},
    )

    build_llm(cfg)

    assert (_CaptureOllama.last_kwargs or {})["context_window"] == 8_192


@pytest.mark.unit
def test_build_llm_ollama_cloud_uses_host_aware_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ollama_mod = ModuleType("llama_index.llms.ollama")

    class _CaptureOllama:
        last_kwargs: dict[str, object] | None = None

        def __init__(self, **kwargs: object) -> None:
            type(self).last_kwargs = dict(kwargs)

    ollama_mod.Ollama = _CaptureOllama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_index.llms.ollama", ollama_mod)

    cfg = DocMindSettings(
        llm_backend="ollama",
        ollama_base_url="https://ollama.com",
        ollama_api_key=SecretStr("ollama-key"),
        openai={
            "api_key": "openai-key",
            "default_headers": {"X-OpenAI": "must-not-leak"},
        },
        security={"allow_remote_endpoints": True},
    )

    build_llm(cfg)

    assert (_CaptureOllama.last_kwargs or {})["headers"] == {
        "authorization": "Bearer ollama-key"
    }


@pytest.mark.unit
def test_build_llm_openai_compatible_responses_uses_openai_responses() -> None:
    """OpenAI-compatible backend uses OpenAIResponses when api_mode=responses."""
    cfg = DocMindSettings()
    cfg.llm_backend = "openai_compatible"
    cfg.llm_request.model = "qwen2.5-7b-instruct"
    cfg.llm_request.context_window = 8192
    cfg.llm_request.max_output_tokens = 1024
    cfg.llm_request.temperature = 0.4
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
        assert kwargs["model"] == cfg.llm_request.model
        assert kwargs["api_key"] == cfg.openai.api_key.get_secret_value()
        assert kwargs["default_headers"] == {"X-Test": "1"}
        assert kwargs["context_window"] == cfg.llm_request.context_window
        assert kwargs["max_output_tokens"] == cfg.llm_request.max_output_tokens
        assert kwargs["temperature"] == pytest.approx(cfg.llm_request.temperature)
        assert float(kwargs["timeout"]) == float(cfg.llm_request_timeout_seconds)
        assert kwargs["max_retries"] == cfg.agents.max_retries


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


def test_vllm_request_config_and_api_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical request settings and vLLM URL reach the adapter."""

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
        llm_request={"model": "Override-Model", "context_window": 8192},
        vllm_base_url="http://localhost:8000",
        llm_request_timeout_seconds=42,
    )

    llm = build_llm(cfg)
    assert str(getattr(llm, "api_base", None)).endswith("/v1")
    assert getattr(llm, "model", None) == "Override-Model"
    assert getattr(llm, "context_window", None) == 8192
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
        llm_request={"context_window": 8192},
        llm_request_timeout_seconds=17,
    )
    llm = build_llm(cfg)
    assert getattr(llm, "api_base", None) == "http://localhost:8080/v1"
    assert getattr(llm, "context_window", None) == 8192
    assert getattr(llm, "timeout", None) == 17.0


def test_invalid_context_window_raises_value_error() -> None:
    """Invalid context_window raises ValueError via validation."""
    with pytest.raises(ValueError, match=r"greater than or equal to 8192"):
        DocMindSettings(llm_request={"context_window": 0})
    with pytest.raises(ValueError, match=r"greater than or equal to 8192"):
        DocMindSettings(llm_request={"context_window": -1024})
