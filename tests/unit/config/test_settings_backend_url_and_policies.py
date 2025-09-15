"""Unit tests for backend base URL normalization and policies in settings."""

from __future__ import annotations

from src.config.settings import DocMindSettings


def test_ensure_v1_normalization_on_backends(monkeypatch):  # type: ignore[no-untyped-def]
    s = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    # vLLM base URL normalized to include /v1
    monkeypatch.setattr(s, "llm_backend", "vllm", raising=False)
    s.vllm_base_url = "http://localhost:8000"  # type: ignore[assignment]
    assert s.backend_base_url_normalized.endswith("/v1")

    # LM Studio already has /v1
    monkeypatch.setattr(s, "llm_backend", "lmstudio", raising=False)
    s.lmstudio_base_url = "http://localhost:1234/v1"  # type: ignore[assignment]
    assert s.backend_base_url_normalized.endswith("/v1")

    # Ollama returns raw base
    monkeypatch.setattr(s, "llm_backend", "ollama", raising=False)
    s.ollama_base_url = "http://localhost:11434"  # type: ignore[assignment]
    assert s.backend_base_url_normalized == "http://localhost:11434"


def test_allow_remote_effective_env_override(monkeypatch):  # type: ignore[no-untyped-def]
    s = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    # Security policy reads centralized security settings
    s.security.allow_remote_endpoints = False
    assert s.allow_remote_effective() is False


def test_get_vllm_config_contains_expected_keys():  # type: ignore[no-untyped-def]
    s = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    cfg = s.get_vllm_config()
    assert set(cfg.keys()) >= {"model", "context_window", "temperature"}
