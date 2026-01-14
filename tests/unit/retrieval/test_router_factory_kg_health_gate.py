"""Test KG tool gating when GraphRAG construction fails."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.retrieval import router_factory as rf

from .conftest import get_router_tool_names


@pytest.mark.unit
def test_kg_tool_absent_when_builder_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graph tool is omitted when build_graph_query_engine raises."""

    def _broken_graph_builder(*_a: object, **_k: object) -> None:
        raise ValueError("broken graph index")

    monkeypatch.setattr(
        rf,
        "build_graph_query_engine",
        _broken_graph_builder,
        raising=True,
    )

    class _Vec:
        def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(qe=True, kwargs=kwargs)

    class _Pg:
        def __init__(self) -> None:
            self.property_graph_store = object()

    cfg = SimpleNamespace(
        enable_graphrag=True,
        retrieval=SimpleNamespace(
            top_k=3,
            use_reranking=False,
            reranking_top_k=2,
            enable_server_hybrid=False,
            enable_image_retrieval=True,
        ),
        graphrag_cfg=SimpleNamespace(default_path_depth=1),
        database=SimpleNamespace(qdrant_collection="col"),
    )

    router = rf.build_router_engine(
        _Vec(), pg_index=_Pg(), settings=cfg, enable_hybrid=False
    )
    assert set(get_router_tool_names(router)) == {
        "semantic_search",
        "multimodal_search",
    }


@pytest.mark.unit
def test_kg_tool_present_when_builder_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graph tool is present when build_graph_query_engine succeeds."""
    monkeypatch.setattr(
        rf,
        "build_graph_query_engine",
        lambda *_a, **_k: SimpleNamespace(query_engine=SimpleNamespace(kind="kg")),
        raising=True,
    )

    class _Vec:
        def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(qe=True, kwargs=kwargs)

    class _Pg:
        def __init__(self) -> None:
            self.property_graph_store = object()

    cfg = SimpleNamespace(
        enable_graphrag=True,
        retrieval=SimpleNamespace(
            top_k=3,
            use_reranking=False,
            reranking_top_k=2,
            enable_server_hybrid=False,
            enable_image_retrieval=True,
        ),
        graphrag_cfg=SimpleNamespace(default_path_depth=1),
        database=SimpleNamespace(qdrant_collection="col"),
    )

    router = rf.build_router_engine(
        _Vec(), pg_index=_Pg(), settings=cfg, enable_hybrid=False
    )
    assert set(get_router_tool_names(router)) == {
        "semantic_search",
        "multimodal_search",
        "knowledge_graph",
    }
