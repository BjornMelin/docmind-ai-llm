"""Offline coverage for router_factory when llama_index is unavailable."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.retrieval.llama_index_adapter import (
    MissingLlamaIndexError,
    llama_index_available,
    set_llama_index_adapter,
)
from src.retrieval.router_factory import GRAPH_DEPENDENCY_HINT, build_router_engine


class _VecIndex:
    def as_query_engine(self, **_kwargs):  # type: ignore[no-untyped-def]
        return MagicMock(name="vector_qe")


def _tool_count(router) -> int:  # type: ignore[no-untyped-def]
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is not None:
            return len(list(tools))
    return 0


def test_router_engine_requires_llama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise a clear error if llama_index is absent and no stub is provided."""
    if llama_index_available():
        pytest.skip(
            "llama_index is installed; missing-dependency scenario not applicable"
        )
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


def test_router_engine_logs_when_graph_disabled() -> None:
    """Emit a warning when GraphRAG is requested without dependency support."""
    vec = _VecIndex()

    class _PgIndex:
        def __init__(self) -> None:
            self.property_graph_store = object()

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

    adapter = SimpleNamespace(
        RouterQueryEngine=SimpleNamespace,
        RetrieverQueryEngine=SimpleNamespace,
        QueryEngineTool=lambda query_engine, metadata: SimpleNamespace(
            query_engine=query_engine, metadata=metadata
        ),
        ToolMetadata=lambda name, description: SimpleNamespace(
            name=name, description=description
        ),
        LLMSingleSelector=SimpleNamespace(
            from_defaults=lambda llm=None: SimpleNamespace(llm=llm)
        ),
        get_pydantic_selector=lambda _llm: None,
        __is_stub__=True,
        supports_graphrag=False,
        graphrag_disabled_reason=GRAPH_DEPENDENCY_HINT,
    )

    set_llama_index_adapter(adapter)
    from loguru import logger

    captured: list[str] = []
    token = logger.add(
        lambda message: captured.append(message.rstrip("\n")), level="WARNING"
    )
    try:
        router = build_router_engine(vec, _PgIndex(), settings=cfg)
    finally:
        logger.remove(token)
        set_llama_index_adapter(None)

    assert _tool_count(router) == 1
    assert any(GRAPH_DEPENDENCY_HINT in message for message in captured)
