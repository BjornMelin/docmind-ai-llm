"""Unit tests for direct LlamaIndex GraphRAG construction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.retrieval import graph_config

pytestmark = [pytest.mark.unit, pytest.mark.requires_llama]


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, query: str) -> list[SimpleNamespace]:
        self.calls.append(query)
        return [SimpleNamespace(node=SimpleNamespace(id_="node-1"))]


class _FakePropertyGraphIndex:
    def __init__(self) -> None:
        self.retriever = _FakeRetriever()
        self.kwargs: dict[str, Any] = {}

    def as_retriever(self, **kwargs: Any) -> _FakeRetriever:
        self.kwargs = kwargs
        return self.retriever


class _FakeQueryEngine:
    @classmethod
    def from_args(cls, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)


@pytest.fixture(autouse=True)
def _stub_llama_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph_config, "RetrieverQueryEngine", _FakeQueryEngine)


def test_build_graph_query_engine_uses_property_graph_index() -> None:
    index = _FakePropertyGraphIndex()
    postprocessor = object()

    artifacts = graph_config.build_graph_query_engine(
        index,
        llm="stub",
        path_depth=2,
        similarity_top_k=5,
        node_postprocessors=[postprocessor],
    )

    assert artifacts.retriever is index.retriever
    assert artifacts.query_engine.retriever is index.retriever
    assert artifacts.query_engine.node_postprocessors == [postprocessor]
    assert index.kwargs == {
        "include_text": True,
        "similarity_top_k": 5,
        "path_depth": 2,
    }


def test_build_graph_retriever_uses_property_graph_index_directly() -> None:
    index = _FakePropertyGraphIndex()

    retriever = graph_config.build_graph_retriever(
        index,
        include_text=False,
        path_depth=3,
        similarity_top_k=7,
    )

    assert retriever is index.retriever
    assert index.kwargs == {
        "include_text": False,
        "similarity_top_k": 7,
        "path_depth": 3,
    }


def test_get_export_seed_ids_uses_graph_retriever() -> None:
    index = _FakePropertyGraphIndex()

    seeds = graph_config.get_export_seed_ids(index, None, cap=2)

    assert seeds == ["node-1"]
    assert index.retriever.calls == ["seed"]
