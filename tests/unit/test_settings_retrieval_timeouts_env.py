"""Unit test for retrieval timeouts settings defaults and env overrides.

Ensures new timeout fields have sane defaults and can be overridden by env.
"""

from __future__ import annotations

from src.config.settings import DocMindSettings


def test_retrieval_timeouts_defaults_and_env(monkeypatch):  # type: ignore[no-untyped-def]
    """Assert defaults and env override behavior for timeout fields."""
    # Defaults via clean instance (no .env)
    s = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    assert s.retrieval.text_rerank_timeout_ms >= 50
    assert s.retrieval.siglip_timeout_ms >= 25
    assert s.retrieval.colpali_timeout_ms >= 25
    assert s.retrieval.total_rerank_budget_ms >= 100

    # Set environment overrides (nested env mapping)
    monkeypatch.setenv("DOCMIND_RETRIEVAL__TEXT_RERANK_TIMEOUT_MS", "777")
    monkeypatch.setenv("DOCMIND_RETRIEVAL__SIGLIP_TIMEOUT_MS", "333")
    monkeypatch.setenv("DOCMIND_RETRIEVAL__COLPALI_TIMEOUT_MS", "444")
    monkeypatch.setenv("DOCMIND_RETRIEVAL__TOTAL_RERANK_BUDGET_MS", "1234")

    # New instance picks up env overrides
    s2 = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    assert s2.retrieval.text_rerank_timeout_ms == 777
    assert s2.retrieval.siglip_timeout_ms == 333
    assert s2.retrieval.colpali_timeout_ms == 444
    assert s2.retrieval.total_rerank_budget_ms == 1234
