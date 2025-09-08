"""Tests for visual reranker factory behavior under failure conditions."""

import pytest

from src.retrieval import reranking as rr


def test_build_visual_reranker_wraps_import_errors(monkeypatch):
    """If ColPaliRerank init raises ImportError, factory raises ValueError."""

    class _BoomError(Exception):
        pass

    def _raise(*_a, **_k):
        raise ImportError("no colpali")

    monkeypatch.setattr(rr, "_build_visual_reranker_cached", _raise)

    with pytest.raises(ValueError, match="ColPaliRerank initialization failed"):
        rr.build_visual_reranker(5)
