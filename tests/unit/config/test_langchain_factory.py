"""Unit tests for LangChain chat model factory (LangGraph runtime)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config.langchain_factory import build_chat_model
from src.config.settings import DocMindSettings

pytestmark = pytest.mark.unit


def test_build_chat_model_default_ollama_uses_effective_model() -> None:
    """Default Ollama wiring uses the public Ollama tag, not the vLLM ID."""
    cfg = DocMindSettings(llm_backend="ollama")

    with patch("src.config.langchain_factory.ChatOpenAI", autospec=True) as client:
        build_chat_model(cfg)

    _, kwargs = client.call_args
    assert kwargs["model"] == "qwen3:4b-instruct"
    assert kwargs["api_key"].get_secret_value() == "not-needed"
    assert kwargs["default_headers"] is None


def test_build_chat_model_openai_compatible_responses_sets_flags() -> None:
    """Responses mode enables use_responses_api and responses/v1 output version."""
    cfg = DocMindSettings()
    cfg.llm_backend = "openai_compatible"
    cfg.openai.base_url = "http://localhost:8000"
    cfg.openai.api_key = "abc"
    cfg.openai.api_mode = "responses"
    cfg.openai.default_headers = {"X-Test": "1"}
    cfg.llm_request.model = "model-1"

    with patch("src.config.langchain_factory.ChatOpenAI", autospec=True) as p:
        inst = MagicMock(name="ChatOpenAIInstance")
        p.return_value = inst
        out = build_chat_model(cfg)
        assert out is inst
        _, kwargs = p.call_args
        assert str(kwargs["base_url"]).endswith("/v1")
        assert kwargs["model"] == "model-1"
        assert kwargs["default_headers"] == {"X-Test": "1"}
        assert kwargs["use_responses_api"] is True
        assert kwargs["output_version"] == "responses/v1"


def test_build_chat_model_openai_compatible_chat_completions_is_default() -> None:
    """Chat-completions mode keeps responses flags disabled."""
    cfg = DocMindSettings()
    cfg.llm_backend = "openai_compatible"
    cfg.openai.base_url = "http://localhost:8000"
    cfg.openai.api_key = "abc"
    cfg.openai.api_mode = "chat_completions"
    cfg.llm_request.model = "model-1"

    with patch("src.config.langchain_factory.ChatOpenAI", autospec=True) as p:
        inst = MagicMock(name="ChatOpenAIInstance")
        p.return_value = inst
        out = build_chat_model(cfg)
        assert out is inst
        _, kwargs = p.call_args
        assert kwargs["use_responses_api"] is False
        assert kwargs["output_version"] is None


def test_build_chat_model_ollama_cloud_uses_only_ollama_auth() -> None:
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

    with patch("src.config.langchain_factory.ChatOpenAI", autospec=True) as client:
        build_chat_model(cfg)

    _, kwargs = client.call_args
    assert kwargs["api_key"].get_secret_value() == "ollama-key"
    assert kwargs["default_headers"] == {"authorization": "Bearer ollama-key"}


def test_build_chat_model_local_ollama_ignores_all_configured_keys() -> None:
    cfg = DocMindSettings(
        llm_backend="ollama",
        ollama_api_key=SecretStr("cloud-key"),
        openai={
            "api_key": "openai-key",
            "default_headers": {"X-OpenAI": "must-not-leak"},
        },
    )

    with patch("src.config.langchain_factory.ChatOpenAI", autospec=True) as client:
        build_chat_model(cfg)

    _, kwargs = client.call_args
    assert kwargs["api_key"].get_secret_value() == "not-needed"
    assert kwargs["default_headers"] is None


@pytest.mark.parametrize("backend", ["vllm", "lmstudio", "llamacpp"])
def test_dedicated_local_backends_do_not_inherit_openai_credentials(
    backend: str,
) -> None:
    cfg = DocMindSettings(
        llm_backend=backend,
        llamacpp_base_url="http://localhost:8080/v1",
        openai={
            "api_key": "openai-key",
            "default_headers": {"X-OpenAI": "must-not-leak"},
        },
    )

    with patch("src.config.langchain_factory.ChatOpenAI", autospec=True) as client:
        build_chat_model(cfg)

    _, kwargs = client.call_args
    assert kwargs["api_key"].get_secret_value() == "not-needed"
    assert kwargs["default_headers"] is None


@pytest.mark.parametrize("backend", ["ollama", "lmstudio", "llamacpp"])
def test_legacy_compatible_backends_send_native_max_tokens_payload(
    backend: str,
) -> None:
    captured: list[dict[str, object]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    cfg = DocMindSettings(
        llm_backend=backend,
        llm_request={"model": "test-model", "max_output_tokens": 137},
        llamacpp_base_url="http://localhost:8080/v1",
    )

    def _real_client(**kwargs: object) -> ChatOpenAI:
        return ChatOpenAI(**kwargs, http_client=client)

    with patch("src.config.langchain_factory.ChatOpenAI", side_effect=_real_client):
        model = build_chat_model(cfg)
    model.invoke("hello")

    assert captured[0]["max_tokens"] == 137
    assert "max_completion_tokens" not in captured[0]
