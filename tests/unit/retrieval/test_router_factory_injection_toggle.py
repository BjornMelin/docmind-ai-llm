"""RouterFactory rerank injection toggle tests.

Validates that node_postprocessors are injected when reranking is enabled and
omitted otherwise.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval import router_factory as rf

from .conftest import get_router_tool_names


class _FakeVector:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        return MagicMock(name="vector_qe")

    def as_retriever(self, **_kwargs):  # type: ignore[no-untyped-def]
        retriever = MagicMock(name="vector_retriever")
        retriever._embed_model = None
        return retriever


class _FakePG:
    def __init__(self) -> None:
        self.property_graph_store = object()


@pytest.mark.unit
def test_router_factory_injects_postprocessors_toggle(
    monkeypatch: pytest.MonkeyPatch,
    router_settings,
) -> None:  # type: ignore[no-untyped-def]
    captured_graph: dict[str, object] = {}

    class _FakeRetrieverQueryEngine:
        @classmethod
        def from_args(cls, **kwargs: object) -> MagicMock:
            return MagicMock(name="multimodal_qe")

    def _fake_build_graph_query_engine(pg_index, **kwargs):  # type: ignore[no-untyped-def]
        del pg_index
        captured_graph.update(kwargs)
        return MagicMock(query_engine=MagicMock(name="graph_qe"))

    def _fake_get_postprocessors(_kind: str, *, use_reranking: bool, **_kwargs: object):
        return ["pp"] if use_reranking else None

    monkeypatch.setattr(
        rf,
        "build_graph_query_engine",
        _fake_build_graph_query_engine,
        raising=True,
    )
    monkeypatch.setattr(
        rf,
        "RetrieverQueryEngine",
        _FakeRetrieverQueryEngine,
        raising=True,
    )
    monkeypatch.setattr(
        rf,
        "get_postprocessors",
        _fake_get_postprocessors,
        raising=True,
    )
    monkeypatch.setattr(
        "src.retrieval.multimodal_fusion.MultimodalFusionRetriever",
        lambda **_kwargs: object(),
        raising=True,
    )

    router_settings.retrieval.use_reranking = True
    router_settings.retrieval.enable_image_retrieval = True

    vec = _FakeVector()
    pg = _FakePG()
    router = rf.build_router_engine(vec, pg_index=pg, settings=router_settings)
    tool_names = set(get_router_tool_names(router))
    assert tool_names == {"semantic_search", "multimodal_search", "knowledge_graph"}
    assert (vec.kwargs or {}).get("node_postprocessors") == ["pp"]
    assert captured_graph.get("node_postprocessors") == ["pp"]

    router_settings.retrieval.use_reranking = False
    captured_graph.clear()
    vec2 = _FakeVector()
    router2 = rf.build_router_engine(vec2, pg_index=_FakePG(), settings=router_settings)
    tool_names2 = set(get_router_tool_names(router2))
    assert tool_names2 == {"semantic_search", "multimodal_search", "knowledge_graph"}
    assert (vec2.kwargs or {}).get("node_postprocessors") is None
    assert captured_graph.get("node_postprocessors") is None
