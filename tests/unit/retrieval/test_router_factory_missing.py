"""Offline coverage for router_factory when llama_index is unavailable."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.retrieval.llama_index_adapter import (
    MissingLlamaIndexError,
    set_llama_index_adapter,
)
from src.retrieval.router_factory import GRAPH_DEPENDENCY_HINT, build_router_engine

from .conftest import get_router_tool_names


class _VecIndex:
    def as_query_engine(self, **_kwargs):  # type: ignore[no-untyped-def]
        return MagicMock(name="vector_qe")


@pytest.mark.unit
def test_router_engine_requires_llama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise a clear error if llama_index is absent and no stub is provided."""
    set_llama_index_adapter(None)
    monkeypatch.setattr(
        "src.retrieval.llama_index_adapter.llama_index_available", lambda: False
    )

    def _raise_import_error() -> None:
        raise ImportError("boom")

    monkeypatch.setattr(
        "src.retrieval.llama_index_adapter._build_real_adapter",
        _raise_import_error,
    )

    cfg = SimpleNamespace(
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
        build_router_engine(_VecIndex(), pg_index=None, settings=cfg)
    assert "pip install docmind_ai_llm[llama]" in str(exc_info.value)


@pytest.mark.unit
def test_router_engine_warns_when_graph_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emit a warning when GraphRAG is requested without dependency support."""
    from src.retrieval.adapter_registry import MissingGraphAdapterError

    class _PgIndex:
        def __init__(self) -> None:
            self.property_graph_store = object()

    monkeypatch.setattr(
        "src.retrieval.router_factory.build_graph_query_engine",
        lambda *_a, **_k: (_ for _ in ()).throw(
            MissingGraphAdapterError(GRAPH_DEPENDENCY_HINT)
        ),
        raising=True,
    )

    cfg = SimpleNamespace(
        enable_graphrag=True,
        retrieval=SimpleNamespace(
            top_k=3,
            use_reranking=False,
            enable_server_hybrid=False,
            reranking_top_k=None,
        ),
        graphrag_cfg=SimpleNamespace(default_path_depth=1),
        database=SimpleNamespace(qdrant_collection="col"),
    )

    from loguru import logger

    captured: list[str] = []
    token = logger.add(
        lambda message: captured.append(message.rstrip("\n")), level="WARNING"
    )
    try:
        router = build_router_engine(_VecIndex(), pg_index=_PgIndex(), settings=cfg)
    finally:
        logger.remove(token)

    assert get_router_tool_names(router) == ["semantic_search"]
    assert any(
        "GraphRAG is disabled" in message or GRAPH_DEPENDENCY_HINT in message
        for message in captured
    )
