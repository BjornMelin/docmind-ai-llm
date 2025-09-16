"""Unit tests for backend base URL normalization and policies in settings."""

from __future__ import annotations

import pytest

from src.config.settings import DocMindSettings


@pytest.mark.parametrize(
    ("backend", "attr", "value", "expect_suffix"),
    [
        ("vllm", "vllm_base_url", "http://localhost:8000", "/v1"),
        ("lmstudio", "lmstudio_base_url", "http://localhost:1234/v1", "/v1"),
        ("ollama", "ollama_base_url", "http://localhost:11434", None),
    ],
)
def test_backend_base_url_normalization(
    monkeypatch, backend: str, attr: str, value: str, expect_suffix: str | None
):  # type: ignore[no-untyped-def]
    settings_obj = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    monkeypatch.setattr(settings_obj, "llm_backend", backend, raising=False)
    setattr(settings_obj, attr, value)
    normalized = settings_obj.backend_base_url_normalized
    if expect_suffix is None:
        assert normalized == value
        return

    assert normalized is not None
    assert normalized.endswith(expect_suffix)


def test_allow_remote_effective_env_override(monkeypatch):  # type: ignore[no-untyped-def]
    s = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    # Security policy reads centralized security settings
    s.security.allow_remote_endpoints = False
    assert s.allow_remote_effective() is False


def test_get_vllm_config_contains_expected_keys():  # type: ignore[no-untyped-def]
    s = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    cfg = s.get_vllm_config()
    assert set(cfg.keys()) >= {"model", "context_window", "temperature"}
