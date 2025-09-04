"""Tests for new settings fields and enforced caps (ADR-024/004)."""

from src.config import settings


def test_reranker_mode_default_present() -> None:
    """Settings include reranker_mode with a valid default."""
    assert hasattr(settings.retrieval, "reranker_mode")
    assert settings.retrieval.reranker_mode in {"auto", "text", "multimodal"}


def test_llm_context_window_max_enforced_default() -> None:
    """Global context cap default is 128K tokens."""
    assert settings.llm_context_window_max == 131072


def test_llm_context_window_enforcement_with_vllm_exceeding_cap(monkeypatch) -> None:
    """If vllm.context_window exceeds the global cap, settings.get_model_config enforces the cap."""
    # Ensure we exceed the cap
    monkeypatch.setattr(
        settings.vllm,
        "context_window",
        int(settings.llm_context_window_max) + 10000,
        raising=True,
    )
    cfg = settings.get_model_config()
    assert cfg["context_window"] == settings.llm_context_window_max


def test_llm_context_window_enforcement_with_vllm_exceeding_cap(monkeypatch) -> None:
    """If vLLM context exceeds cap, get_model_config enforces the cap."""
    monkeypatch.setattr(
        settings.vllm, "context_window", settings.llm_context_window_max + 1000
    )
    config = settings.get_model_config()
    assert config["context_window"] == settings.llm_context_window_max
