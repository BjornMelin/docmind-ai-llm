"""Unit tests for the LlamaIndex adapter factory."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from src.retrieval.adapters import GraphQueryArtifacts
from src.retrieval.llama_index_adapter import (
    LlamaIndexAdapterFactory,
    MissingLlamaIndexError,
    build_llama_index_factory,
)


class _FakeRetriever:
    def __init__(self, storage_context: object, **kwargs: object) -> None:
        self.storage_context = storage_context
        self.kwargs = kwargs

    def retrieve(self, _query: str, /, *args: object, **kwargs: object) -> list[int]:
        return [1, 2]


class _FakeRetrieverQueryEngine:
    @classmethod
    def from_args(cls, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)


class _FakePropertyGraphIndex:
    def __init__(self) -> None:
        self.storage_context = object()
        self.property_graph_store = SimpleNamespace()


@pytest.fixture(name="mock_llamaindex_modules")
def _mock_llamaindex_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch importlib to provide fake llama_index modules."""

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "llama_index.core.retrievers":
            return SimpleNamespace(KnowledgeGraphRAGRetriever=_FakeRetriever)
        if name == "llama_index.core.query_engine":
            return SimpleNamespace(RetrieverQueryEngine=_FakeRetrieverQueryEngine)
        if name == "llama_index.core.indices.property_graph":
            return SimpleNamespace(PropertyGraphIndex=_FakePropertyGraphIndex)
        if name == "llama_index.core.selectors":
            return SimpleNamespace(
                LLMSingleSelector=SimpleNamespace(
                    from_defaults=lambda llm=None: SimpleNamespace(llm=llm)
                )
            )
        if name == "llama_index.core.tools":
            return SimpleNamespace(
                QueryEngineTool=lambda **kwargs: SimpleNamespace(**kwargs),
                ToolMetadata=lambda **kwargs: SimpleNamespace(**kwargs),
            )
        raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", _fake_import)
    monkeypatch.setattr(
        "src.retrieval.llama_index_adapter.metadata.version",
        lambda _: "0.10.1",
    )


def test_build_factory_success(mock_llamaindex_modules: None) -> None:
    """Factory initializes when dependencies are resolvable."""
    factory = build_llama_index_factory()
    assert isinstance(factory, LlamaIndexAdapterFactory)
    assert factory.supports_graphrag is True
    assert factory.version == "0.10.1"


def test_graph_artifacts_construction(mock_llamaindex_modules: None) -> None:
    """Factory returns GraphQueryArtifacts with retriever and query engine."""
    factory = build_llama_index_factory()
    pg_index = _FakePropertyGraphIndex()
    artifacts = factory.build_graph_artifacts(property_graph_index=pg_index)
    assert isinstance(artifacts, GraphQueryArtifacts)
    assert hasattr(artifacts.retriever, "retrieve")
    assert hasattr(artifacts.query_engine, "retriever")


def test_missing_dependency_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing modules surface ``MissingLlamaIndexError``."""
    monkeypatch.setattr(
        "importlib.import_module",
        mock.Mock(side_effect=ImportError("llama_index")),
    )
    monkeypatch.setattr(
        "src.retrieval.llama_index_adapter.metadata.version",
        lambda _: "0.10.1",
    )
    with pytest.raises(MissingLlamaIndexError):
        build_llama_index_factory()
