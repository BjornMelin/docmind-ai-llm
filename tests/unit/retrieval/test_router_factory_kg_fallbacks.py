"""RouterFactory KG fallbacks when node_postprocessors are unsupported."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.retrieval import router_factory as rf

from .conftest import get_router_tool_names


@pytest.mark.unit
def test_router_factory_retries_kg_without_postprocessors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph tool construction retries without node_postprocessors on TypeError."""
    calls: list[object] = []

    def _fake_build_graph_query_engine(_pg, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs.get("node_postprocessors"))
        if kwargs.get("node_postprocessors") is not None:
            raise TypeError("node_postprocessors unsupported")
        return SimpleNamespace(query_engine=SimpleNamespace(kind="kg"))

    monkeypatch.setattr(
        rf,
        "build_graph_query_engine",
        _fake_build_graph_query_engine,
        raising=True,
    )
    monkeypatch.setattr(
        "src.retrieval.reranking.get_postprocessors",
        lambda *_a, **_k: ["pp"],
        raising=False,
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
            top_k=3, use_reranking=True, reranking_top_k=2, enable_server_hybrid=False
        ),
        graphrag_cfg=SimpleNamespace(default_path_depth=1),
        database=SimpleNamespace(qdrant_collection="col"),
    )

    router = rf.build_router_engine(
        _Vec(), pg_index=_Pg(), settings=cfg, enable_hybrid=False
    )
    assert "knowledge_graph" in get_router_tool_names(router)
    assert calls == [["pp"], None]
