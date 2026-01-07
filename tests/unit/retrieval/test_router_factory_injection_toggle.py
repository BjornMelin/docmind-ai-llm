"""RouterFactory rerank injection toggle tests.

Validates that node_postprocessors are injected when reranking is enabled and
omitted otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.retrieval import router_factory as rf

from .conftest import get_router_tool_names


class _FakeVector:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        return SimpleNamespace(qe=True, kwargs=kwargs)


class _FakePG:
    def __init__(self) -> None:
        self.property_graph_store = object()


@pytest.mark.unit
def test_router_factory_injects_postprocessors_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_graph: dict[str, object] = {}

    def _fake_build_graph_query_engine(pg_index, **kwargs):  # type: ignore[no-untyped-def]
        del pg_index
        captured_graph.update(kwargs)
        return SimpleNamespace(query_engine=SimpleNamespace())

    def _fake_get_postprocessors(_kind: str, *, use_reranking: bool, **_kwargs: object):
        return ["pp"] if use_reranking else None

    monkeypatch.setattr(
        "src.retrieval.router_factory.build_graph_query_engine",
        _fake_build_graph_query_engine,
        raising=True,
    )
    monkeypatch.setattr(
        "src.retrieval.reranking.get_postprocessors",
        _fake_get_postprocessors,
        raising=True,
    )

    cfg = SimpleNamespace(
        enable_graphrag=True,
        retrieval=SimpleNamespace(
            top_k=5, use_reranking=True, reranking_top_k=3, enable_server_hybrid=False
        ),
        graphrag_cfg=SimpleNamespace(default_path_depth=1),
        database=SimpleNamespace(qdrant_collection="col"),
    )

    vec = _FakeVector()
    pg = _FakePG()
    router = rf.build_router_engine(vec, pg_index=pg, settings=cfg, enable_hybrid=False)
    assert len(get_router_tool_names(router)) == 2
    assert vec.kwargs.get("node_postprocessors") == ["pp"]
    assert captured_graph.get("node_postprocessors") == ["pp"]

    cfg.retrieval.use_reranking = False
    captured_graph.clear()
    vec2 = _FakeVector()
    router2 = rf.build_router_engine(
        vec2, pg_index=_FakePG(), settings=cfg, enable_hybrid=False
    )
    assert len(get_router_tool_names(router2)) == 2
    assert vec2.kwargs.get("node_postprocessors") is None
    assert captured_graph.get("node_postprocessors") is None
