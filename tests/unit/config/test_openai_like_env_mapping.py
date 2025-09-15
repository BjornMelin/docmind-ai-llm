"""Tests for OpenAI-like environment variable mapping to DocMindSettings."""

import pytest

from src.config.settings import DocMindSettings


@pytest.mark.unit
def test_openai_like_env_mapping(monkeypatch):
    """Test that OpenAI-like environment variables are correctly mapped to settings."""
    monkeypatch.setenv("DOCMIND_OPENAI__API_KEY", "key-123")
    monkeypatch.setenv("DOCMIND_OPENAI__BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("DOCMIND_LLM_REQUEST_TIMEOUT_SECONDS", "222")
    monkeypatch.setenv("DOCMIND_LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("DOCMIND_VLLM__VLLM_BASE_URL", "http://127.0.0.1:8000")

    cfg = DocMindSettings()
    assert cfg.openai.api_key == "key-123"
    assert cfg.openai.base_url.endswith("/v1")
    assert cfg.llm_request_timeout_seconds == 222
    assert cfg.lmstudio_base_url.endswith("/v1")
    assert cfg.vllm.vllm_base_url == "http://127.0.0.1:8000"
