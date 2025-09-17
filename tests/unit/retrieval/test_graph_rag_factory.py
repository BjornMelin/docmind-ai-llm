"""Tests for graph query/retriever helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.retrieval import graph_config
from src.retrieval.graph_config import (
    GraphQueryArtifacts,
    build_graph_query_engine,
    build_graph_retriever,
)


class _StubIndex:
    """Stub ``PropertyGraphIndex`` exposing ``as_retriever``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, int | bool]] = []

    def as_retriever(
        self, include_text: bool = True, similarity_top_k: int = 10, path_depth: int = 1
    ) -> dict[str, int | bool]:
        self.calls.append(
            {
                "include_text": include_text,
                "similarity_top_k": similarity_top_k,
                "path_depth": path_depth,
            }
        )
        return {"retriever": True}


@pytest.mark.unit
def test_build_graph_retriever_returns_expected() -> None:
    """Helper delegates to PropertyGraphIndex.as_retriever with kwargs."""
    idx = _StubIndex()
    retriever = build_graph_retriever(
        idx, include_text=False, similarity_top_k=7, path_depth=3
    )
    assert retriever["retriever"] is True
    assert idx.calls == [
        {"include_text": False, "similarity_top_k": 7, "path_depth": 3}
    ]


@pytest.mark.unit
def test_build_graph_query_engine_uses_retriever(monkeypatch) -> None:
    """Helper builds RetrieverQueryEngine.from_args with provided kwargs."""
    idx = _StubIndex()

    captured: dict[str, dict[str, object]] = {}

    def _fake_from_args(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(result="ok")

    monkeypatch.setattr(
        graph_config, "RetrieverQueryEngine", SimpleNamespace(from_args=_fake_from_args)
    )

    artifacts = build_graph_query_engine(
        idx,
        llm="stub-llm",
        include_text=False,
        similarity_top_k=5,
        path_depth=2,
        node_postprocessors=["post"],
        response_mode="tree",
    )

    assert isinstance(artifacts, GraphQueryArtifacts)
    assert artifacts.query_engine.result == "ok"
    assert artifacts.retriever["retriever"] is True
    assert captured["kwargs"]["retriever"]["retriever"] is True
    assert captured["kwargs"]["llm"] == "stub-llm"
    assert captured["kwargs"]["response_mode"] == "tree"
    assert captured["kwargs"]["node_postprocessors"] == ["post"]


@pytest.mark.unit
def test_build_graph_query_engine_requires_retriever(monkeypatch) -> None:
    """Helper raises ValueError when LlamaIndex is unavailable."""
    idx = object()
    monkeypatch.setattr(graph_config, "RetrieverQueryEngine", graph_config.Any)
    with pytest.raises(ValueError, match="LlamaIndex is required"):
        build_graph_query_engine(idx)
