"""Unit tests for llama-index adapter utilities and GraphRAG factory."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from src.retrieval.adapters import GraphQueryArtifacts
from src.retrieval.llama_index_adapter import (
    LlamaIndexAdapterFactory,
    MissingLlamaIndexError,
    build_llama_index_factory,
    get_llama_index_adapter,
    llama_index_available,
    set_llama_index_adapter,
)


def test_llama_index_available_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns False when find_spec cannot locate llama_index.core."""
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: None if name == "llama_index.core" else mock.DEFAULT,
    )
    assert llama_index_available() is False


def test_get_adapter_missing_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surface the llama extra install hint when the dependency is absent."""
    set_llama_index_adapter(None)
    monkeypatch.setattr(
        "src.retrieval.llama_index_adapter.llama_index_available",
        lambda: False,
    )
    with pytest.raises(MissingLlamaIndexError) as exc_info:
        get_llama_index_adapter(force_reload=True)
    assert "pip install docmind_ai_llm[llama]" in str(exc_info.value)


class _FakeRetriever:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def retrieve(self, _query: str, /, *args: object, **kwargs: object) -> list[int]:
        return [1, 2]


class _FakeRetrieverQueryEngine:
    @classmethod
    def from_args(cls, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)


class _FakePropertyGraphIndex:
    def as_retriever(self, **_kwargs: object) -> _FakeRetriever:
        return _FakeRetriever(**_kwargs)


@pytest.fixture(name="mock_llamaindex")
def _mock_llamaindex(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch minimal seams so the factory can be instantiated without llama-index."""
    monkeypatch.setattr(
        "src.retrieval.llama_index_adapter.llama_index_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "src.retrieval.llama_index_adapter.metadata.version",
        lambda _dist: "0.13.4",
    )
    fake_adapter = SimpleNamespace(RetrieverQueryEngine=_FakeRetrieverQueryEngine)
    monkeypatch.setattr(
        "src.retrieval.llama_index_adapter.get_llama_index_adapter",
        lambda *_a, **_k: fake_adapter,
    )
    # Avoid importing the real module in _load_graph_modules()

    def _fake_import_module(name: str):
        if name == "llama_index.core.indices.property_graph":
            return SimpleNamespace(PropertyGraphIndex=_FakePropertyGraphIndex)
        raise ImportError(name)

    monkeypatch.setattr(
        "importlib.import_module",
        _fake_import_module,
    )


def test_build_factory_success(mock_llamaindex: None) -> None:
    """Factory initializes when dependencies are resolvable."""
    factory = build_llama_index_factory()
    assert isinstance(factory, LlamaIndexAdapterFactory)
    assert factory.supports_graphrag is True
    assert factory.version == "0.13.4"


def test_graph_artifacts_construction(mock_llamaindex: None) -> None:
    """Factory returns GraphQueryArtifacts with retriever and query engine."""
    factory = build_llama_index_factory()
    pg_index = _FakePropertyGraphIndex()
    artifacts = factory.build_graph_artifacts(property_graph_index=pg_index)
    assert isinstance(artifacts, GraphQueryArtifacts)
    assert hasattr(artifacts.retriever, "retrieve")
    assert hasattr(artifacts.query_engine, "retriever")


def test_graph_factory_missing_dependency_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing graph dependencies surface a GraphRAG-oriented hint."""
    monkeypatch.setattr(
        "src.retrieval.llama_index_adapter.llama_index_available",
        lambda: False,
    )
    with pytest.raises(MissingLlamaIndexError) as exc_info:
        build_llama_index_factory()
    assert "docmind_ai_llm[graph]" in str(exc_info.value)
