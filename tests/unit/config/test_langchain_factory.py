"""Unit tests for LangChain chat model factory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config.langchain_factory import build_chat_model

pytestmark = pytest.mark.unit


def test_build_chat_model_requires_model_name() -> None:
    cfg = SimpleNamespace(
        model=None,
        vllm=SimpleNamespace(model="", temperature=0.0),
        ui=SimpleNamespace(request_timeout_seconds=1.0),
        llm_request_timeout_seconds=None,
        backend_base_url_normalized="http://localhost:8000/v1",
        llm_backend="vllm",
        ollama_base_url="http://localhost:11434",
        openai=SimpleNamespace(api_key=""),
        agents=SimpleNamespace(max_retries=0),
    )
    with pytest.raises(ValueError, match="No model name configured"):
        build_chat_model(cfg)  # type: ignore[arg-type]


def test_build_chat_model_requires_base_url() -> None:
    cfg = SimpleNamespace(
        model="m",
        vllm=SimpleNamespace(model="m", temperature=0.0),
        ui=SimpleNamespace(request_timeout_seconds=1.0),
        llm_request_timeout_seconds=1.0,
        backend_base_url_normalized="",
        llm_backend="vllm",
        ollama_base_url="http://localhost:11434",
        openai=SimpleNamespace(api_key=""),
        agents=SimpleNamespace(max_retries=0),
    )
    with pytest.raises(ValueError, match="No backend base URL configured"):
        build_chat_model(cfg)  # type: ignore[arg-type]


def test_build_chat_model_ollama_uses_ensure_v1(monkeypatch) -> None:
    cfg = SimpleNamespace(
        model="m",
        vllm=SimpleNamespace(model="m", temperature=0.0),
        ui=SimpleNamespace(request_timeout_seconds=1.0),
        llm_request_timeout_seconds=1.0,
        backend_base_url_normalized="http://ignored",
        llm_backend="ollama",
        ollama_base_url="http://localhost:11434",
        openai=SimpleNamespace(api_key=""),
        agents=SimpleNamespace(max_retries=0),
    )
    seen: list[str] = []
    monkeypatch.setattr(
        "src.config.langchain_factory.ensure_v1",
        lambda url: seen.append(str(url)) or "http://localhost:11434/v1",
    )
    created: dict[str, object] = {}

    class _ChatOpenAI:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            created.update(kwargs)

    monkeypatch.setattr("src.config.langchain_factory.ChatOpenAI", _ChatOpenAI)
    build_chat_model(cfg)  # type: ignore[arg-type]
    assert seen == ["http://localhost:11434"]
    assert created.get("base_url") == "http://localhost:11434/v1"
