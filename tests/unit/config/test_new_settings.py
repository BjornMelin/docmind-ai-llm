"""Tests for new settings fields and enforced caps (ADR-024/004)."""

from src.config import settings


def test_reranker_mode_default_present() -> None:
    """Settings include reranker_mode with a valid default."""
    assert hasattr(settings.retrieval, "reranker_mode")
    assert settings.retrieval.reranker_mode in {"auto", "text", "multimodal"}


def test_llm_context_window_max_enforced_default() -> None:
    """Global context cap default is 128K tokens."""
    assert settings.llm_context_window_max == 131072
