"""Unit tests for graph configuration helpers with a fake adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from src.retrieval.adapters.protocols import (
    AdapterFactoryProtocol,
    GraphIndexBuilderProtocol,
    GraphQueryArtifacts,
    GraphQueryEngineProtocol,
    GraphRetrieverProtocol,
    TelemetryHooksProtocol,
)
from src.retrieval.graph_config import (
    build_graph_query_engine,
    build_graph_retriever,
    get_export_seed_ids,
)

pytestmark = pytest.mark.requires_llama


@dataclass
class _FakeTelemetry(TelemetryHooksProtocol):
    events: list[str]

    def router_built(
        self,
        *,
        adapter_name: str,
        supports_graphrag: bool,
        tools: Sequence[str],
    ) -> None:
        self.events.append(f"router:{adapter_name}:{supports_graphrag}:{len(tools)}")

    def graph_exported(
        self,
        *,
        adapter_name: str,
        fmt: str,
        bytes_written: int,
        depth: int,
        seed_count: int,
    ) -> None:
        self.events.append(
            f"export:{adapter_name}:{fmt}:{bytes_written}:{depth}:{seed_count}"
        )


@dataclass
class _FakeRetriever(GraphRetrieverProtocol):
    calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]]

    def retrieve(self, query: Any, /, *args: Any, **kwargs: Any) -> Sequence[Any]:
        self.calls.append((query, args, kwargs))
        return [SimpleNamespace(node=SimpleNamespace(id_="node-1"))]


@dataclass
class _FakeEngine(GraphQueryEngineProtocol):
    retriever: _FakeRetriever

    def query(self, query: Any, /, *args: Any, **kwargs: Any) -> Any:
        self.retriever.retrieve(query)
        return "ok"

    async def aquery(self, query: Any, /, *args: Any, **kwargs: Any) -> Any:
        return self.query(query, *args, **kwargs)


@dataclass
class _FakeAdapter(AdapterFactoryProtocol):
    name: str = "fake"
    version: str = "1.0.0"
    supports_graphrag: bool = True
    dependency_hint: str = "install fake"
    retriever: _FakeRetriever = field(default_factory=lambda: _FakeRetriever(calls=[]))
    telemetry: _FakeTelemetry = field(default_factory=lambda: _FakeTelemetry(events=[]))

    def build_graph_artifacts(
        self,
        *,
        property_graph_index: Any,
        llm: Any | None = None,
        include_text: bool = True,
        similarity_top_k: int = 10,
        path_depth: int = 1,
        node_postprocessors: Sequence[Any] | None = None,
        response_mode: str = "compact",
    ) -> GraphQueryArtifacts:
        artifact_engine = _FakeEngine(retriever=self.retriever)
        return GraphQueryArtifacts(
            retriever=self.retriever,
            query_engine=artifact_engine,
            exporter=None,
            telemetry=self.telemetry,
        )

    def get_index_builder(self) -> GraphIndexBuilderProtocol | None:
        return None

    def get_telemetry_hooks(self) -> TelemetryHooksProtocol:
        return self.telemetry


@pytest.fixture
def fake_adapter() -> _FakeAdapter:
    return _FakeAdapter()


@pytest.mark.unit
def test_build_graph_retriever_uses_adapter(fake_adapter: _FakeAdapter) -> None:
    retriever = build_graph_retriever(
        property_graph_index=object(), adapter=fake_adapter
    )
    assert retriever is fake_adapter.retriever
    result = retriever.retrieve("question")
    assert len(result) == 1
    assert fake_adapter.retriever.calls[0][0] == "question"


@pytest.mark.unit
def test_build_graph_query_engine(fake_adapter: _FakeAdapter) -> None:
    artifacts = build_graph_query_engine(
        property_graph_index=object(),
        adapter=fake_adapter,
        llm="stub",
        path_depth=2,
        similarity_top_k=5,
    )
    assert isinstance(artifacts, GraphQueryArtifacts)
    assert artifacts.query_engine.query("test") == "ok"
    assert fake_adapter.retriever.calls[-1][0] == "test"


@pytest.mark.unit
def test_get_export_seed_ids_uses_graph_retriever(fake_adapter: _FakeAdapter) -> None:
    seeds = get_export_seed_ids(object(), None, cap=2, adapter=fake_adapter)
    assert seeds == ["node-1"]
