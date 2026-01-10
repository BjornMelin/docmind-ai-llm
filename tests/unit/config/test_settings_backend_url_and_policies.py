"""Unit tests for backend base URL normalization and policies in settings."""

from __future__ import annotations

import pytest

from src.config.settings import DocMindSettings


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


def test_backend_base_url_openai_group_overrides_when_customized():  # type: ignore[no-untyped-def]
    s = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    s.llm_backend = "vllm"
    s.vllm_base_url = "http://localhost:8000"
    s.openai.base_url = "http://localhost:9999/v1"
    assert s.backend_base_url_normalized == "http://localhost:9999/v1"


def test_allow_remote_effective_env_override(monkeypatch):  # type: ignore[no-untyped-def]
    s = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    # Security policy reads centralized security settings
    s.security.allow_remote_endpoints = False
    assert s.allow_remote_effective() is False


def test_get_vllm_config_contains_expected_keys():  # type: ignore[no-untyped-def]
    s = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    cfg = s.get_vllm_config()
    assert set(cfg.keys()) >= {"model", "context_window", "temperature"}
