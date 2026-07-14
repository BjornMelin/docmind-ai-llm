"""Unit tests for backend base URL normalization and policies in settings."""

from __future__ import annotations

from typing import Literal

import pytest

from src.config.settings import DocMindSettings, LLMRequestConfig


@pytest.mark.parametrize(
    ("backend", "model", "expected"),
    [
        ("ollama", None, "qwen3:4b-instruct"),
        ("vllm", None, "Qwen/Qwen3-4B-Instruct-2507-FP8"),
        ("ollama", "explicit-model", "explicit-model"),
    ],
)
def test_effective_model_is_backend_aware(
    backend: Literal["ollama", "vllm"],
    model: str | None,
    expected: str,
) -> None:
    """Resolve one model identity for every backend-agnostic consumer."""
    settings = DocMindSettings(
        llm_backend=backend,
        llm_request=LLMRequestConfig(model=model),
    )

    assert settings.effective_model == expected


@pytest.mark.parametrize(
    ("backend", "attr", "value", "expected"),
    [
        ("vllm", "vllm_base_url", "http://localhost:8000", "http://localhost:8000/v1"),
        (
            "lmstudio",
            "lmstudio_base_url",
            "http://localhost:1234/v1",
            "http://localhost:1234/v1",
        ),
        (
            "ollama",
            "ollama_base_url",
            "http://localhost:11434",
            "http://localhost:11434",
        ),
    ],
)
def test_backend_base_url_normalization(
    monkeypatch, backend: str, attr: str, value: str, expected: str
):  # type: ignore[no-untyped-def]
    settings_obj = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    monkeypatch.setattr(settings_obj, "llm_backend", backend, raising=False)
    setattr(settings_obj, attr, value)
    normalized = settings_obj.backend_base_url_normalized
    assert normalized == expected


def test_vllm_base_url_has_single_owner():  # type: ignore[no-untyped-def]
    s = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    s.llm_backend = "vllm"
    s.vllm_base_url = "http://localhost:8000"
    s.openai = s.openai.model_copy(update={"base_url": "http://localhost:9999/v1"})
    assert s.backend_base_url_normalized == "http://localhost:8000/v1"


@pytest.mark.parametrize(
    ("backend", "dedicated_field", "dedicated_url"),
    [
        ("lmstudio", "lmstudio_base_url", "http://localhost:1234/v1"),
        ("llamacpp", "llamacpp_base_url", "http://localhost:8080/v1"),
    ],
)
def test_dedicated_backend_url_cannot_be_overridden_by_openai_config(
    backend: Literal["lmstudio", "llamacpp"],
    dedicated_field: str,
    dedicated_url: str,
) -> None:
    """Selecting a local backend cannot reuse a prior cloud endpoint."""
    settings = DocMindSettings(
        llm_backend=backend,
        openai={"base_url": "https://api.openai.com/v1", "api_key": "secret"},
        security={"allow_remote_endpoints": True},
        **{dedicated_field: dedicated_url},
    )

    assert settings.backend_base_url_normalized == dedicated_url


def test_allow_remote_effective_env_override(monkeypatch):  # type: ignore[no-untyped-def]
    s = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    # Security policy reads centralized security settings
    s.security.allow_remote_endpoints = False
    assert s.allow_remote_effective() is False


def test_llm_request_env_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCMIND_LLM_REQUEST__MODEL", "served-model")
    monkeypatch.setenv("DOCMIND_LLM_REQUEST__CONTEXT_WINDOW", "32768")
    monkeypatch.setenv("DOCMIND_LLM_REQUEST__MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("DOCMIND_LLM_REQUEST__TEMPERATURE", "0.3")

    settings = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    assert settings.effective_model == "served-model"
    assert settings.llm_request.context_window == 32768
    assert settings.llm_request.max_output_tokens == 1024
    assert settings.llm_request.temperature == pytest.approx(0.3)
