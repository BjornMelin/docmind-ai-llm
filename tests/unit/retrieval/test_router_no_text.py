"""Retrieval-only router engines must not synthesize unused answers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.config.settings import DocMindSettings
from src.retrieval import graph_config, router_factory

pytestmark = [pytest.mark.unit, pytest.mark.requires_llama]


class _OwnedRetriever:
    def __init__(self, name: str) -> None:
        self.name = name

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


class _VectorIndex:
    def __init__(self) -> None:
        self.query_engine_kwargs: dict[str, object] = {}

    def as_query_engine(self, **kwargs: object) -> MagicMock:
        self.query_engine_kwargs = kwargs
        return MagicMock(name="vector_query_engine")


def test_router_configures_every_retrieval_engine_for_no_text(
    monkeypatch: pytest.MonkeyPatch,
    router_settings: DocMindSettings,
) -> None:
    """Vector and optional retrievers all bypass answer synthesis."""
    response_modes: dict[str, object] = {}

    class _RecordingQueryEngine:
        @classmethod
        def from_args(cls, **kwargs: Any) -> MagicMock:
            retriever = kwargs["retriever"]
            response_modes[retriever.name] = kwargs["response_mode"]
            return MagicMock(name=f"{retriever.name}_query_engine")

    monkeypatch.setattr(router_factory, "RetrieverQueryEngine", _RecordingQueryEngine)
    monkeypatch.setattr(
        "src.retrieval.hybrid.ServerHybridRetriever",
        lambda _params: _OwnedRetriever("hybrid"),
    )
    monkeypatch.setattr(
        "src.retrieval.keyword.KeywordSparseRetriever",
        lambda _params: _OwnedRetriever("keyword"),
    )
    monkeypatch.setattr(
        "src.retrieval.multimodal_fusion.MultimodalFusionRetriever",
        lambda **_kwargs: _OwnedRetriever("multimodal"),
    )
    monkeypatch.setattr(router_factory, "get_postprocessors", lambda *_a, **_k: [])

    router_settings.retrieval.enable_image_retrieval = True
    router_settings.retrieval.enable_server_hybrid = True
    router_settings.retrieval.enable_keyword_tool = True
    vector_index = _VectorIndex()

    router_factory.build_router_engine(
        vector_index,  # type: ignore[arg-type]
        settings=router_settings,
    )

    assert vector_index.query_engine_kwargs["response_mode"] is ResponseMode.NO_TEXT
    assert response_modes == {
        "hybrid": ResponseMode.NO_TEXT,
        "keyword": ResponseMode.NO_TEXT,
        "multimodal": ResponseMode.NO_TEXT,
    }


def test_graph_query_engine_is_retrieval_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GraphRAG construction fixes the native engine to NO_TEXT mode."""
    retriever = object()

    class _GraphIndex:
        def as_retriever(self, **_kwargs: object) -> object:
            return retriever

    class _RecordingQueryEngine:
        @classmethod
        def from_args(cls, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(**kwargs)

    monkeypatch.setattr(graph_config, "RetrieverQueryEngine", _RecordingQueryEngine)

    artifacts = graph_config.build_graph_query_engine(_GraphIndex())

    assert artifacts.retriever is retriever
    assert artifacts.query_engine.response_mode is ResponseMode.NO_TEXT


class _StaticRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        del query_bundle
        return [
            NodeWithScore(
                node=TextNode(id_="retrieved-node", text="retrieved context"),
                score=0.9,
            )
        ]

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self._retrieve(query_bundle)


async def test_no_text_preserves_sources_without_calling_the_llm() -> None:
    """LlamaIndex NO_TEXT returns source nodes without an answer-model call."""
    llm = MockLLM(max_tokens=8)
    engine = RetrieverQueryEngine.from_args(
        retriever=_StaticRetriever(),
        llm=llm,
        response_mode=ResponseMode.NO_TEXT,
    )

    with (
        patch.object(
            MockLLM, "complete", side_effect=AssertionError("LLM called")
        ) as complete,
        patch.object(
            MockLLM, "acomplete", side_effect=AssertionError("LLM called")
        ) as acomplete,
    ):
        response = await engine.aquery("find the source")

    complete.assert_not_called()
    acomplete.assert_not_called()
    assert isinstance(response, Response)
    assert response.response == ""
    assert [source.node.node_id for source in response.source_nodes] == [
        "retrieved-node"
    ]
