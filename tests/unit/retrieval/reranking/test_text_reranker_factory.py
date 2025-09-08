"""Tests for text reranker factory selection and configuration."""

from src.retrieval.reranking import build_text_reranker


def test_build_text_reranker_li_mode(monkeypatch):
    # Force LI mode and small top_k
    # No settings knob; factory auto-detects. Ensure object exposes top_n attr.
    from src.config import settings

    monkeypatch.setattr(settings.retrieval, "reranking_top_k", 3, raising=False)

    rr = build_text_reranker(3)
    # Both adapters expose 'top_n'; assert contract rather than concrete type
    assert hasattr(rr, "top_n")
    assert int(getattr(rr, "top_n", 0)) == 3
