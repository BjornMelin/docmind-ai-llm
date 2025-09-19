"""Offline coverage for router_factory when llama_index is unavailable."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.retrieval.llama_index_adapter import (
    MissingLlamaIndexError,
    set_llama_index_adapter,
)
from src.retrieval.router_factory import build_router_engine


class _VecIndex:
    def as_query_engine(self, **_kwargs):  # type: ignore[no-untyped-def]
        return MagicMock(name="vector_qe")


def test_router_engine_requires_llama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise a clear error if llama_index is absent and no stub is provided."""
    set_llama_index_adapter(None)
    monkeypatch.setattr(
        "src.retrieval.llama_index_adapter.llama_index_available", lambda: False
    )

    def _boom() -> None:
        raise ImportError("boom")

    monkeypatch.setattr(
        "src.retrieval.llama_index_adapter._build_real_adapter",
        _boom,
    )
    settings = SimpleNamespace(
        enable_graphrag=False,
        retrieval=SimpleNamespace(
            top_k=3,
            use_reranking=False,
            enable_server_hybrid=False,
            reranking_top_k=None,
        ),
        database=SimpleNamespace(qdrant_collection="col"),
    )
    with pytest.raises(MissingLlamaIndexError) as exc_info:
        build_router_engine(_VecIndex(), pg_index=None, settings=settings)
    assert "pip install docmind_ai_llm[llama]" in str(exc_info.value)
