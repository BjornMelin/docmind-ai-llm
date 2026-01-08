"""Runtime-checked protocols describing GraphRAG adapter contracts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphIndexBuilderProtocol(Protocol):
    """Factory responsible for producing property graph indices."""

    def build_from_documents(
        self,
        *,
        documents: Iterable[Any],
        storage_context: Any | None = None,
        llm: Any | None = None,
        embed_model: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return a property graph index (e.g. ``PropertyGraphIndex``)."""
        raise NotImplementedError


@runtime_checkable
class GraphRetrieverProtocol(Protocol):
    """Retriever capable of returning graph-aware nodes."""

    def retrieve(self, query: Any, /, *args: Any, **kwargs: Any) -> Sequence[Any]:
        """Return nodes for ``query``."""
        raise NotImplementedError


@runtime_checkable
class GraphQueryEngineProtocol(Protocol):
    """Query engine operating on a graph retriever."""

    def query(self, query: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Execute a synchronous query."""
        raise NotImplementedError

    async def aquery(self, query: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Execute an asynchronous query."""
        raise NotImplementedError


@runtime_checkable
class GraphExporterProtocol(Protocol):
    """Exporter surfaces used for graph persistence."""

    def export_jsonl(
        self,
        *,
        path: Path | str,
        seed_node_ids: Sequence[str],
        depth: int,
    ) -> None:
        """Persist the graph to a JSON Lines file."""
        raise NotImplementedError

    def export_parquet(
        self,
        *,
        path: Path | str,
        seed_node_ids: Sequence[str],
        depth: int,
    ) -> None:
        """Persist the graph to a Parquet file."""
        raise NotImplementedError


@runtime_checkable
class TelemetryHooksProtocol(Protocol):
    """Hook points for emitting observability events."""

    def router_built(
        self,
        *,
        adapter_name: str,
        supports_graphrag: bool,
        tools: Sequence[str],
    ) -> None:
        """Record router construction details."""
        raise NotImplementedError

    def graph_exported(
        self,
        *,
        adapter_name: str,
        fmt: str,
        bytes_written: int,
        depth: int,
        seed_count: int,
    ) -> None:
        """Record graph export metrics."""
        raise NotImplementedError


@dataclass(frozen=True)
class GraphQueryArtifacts:
    """Container describing GraphRAG runtime services."""

    retriever: GraphRetrieverProtocol
    query_engine: GraphQueryEngineProtocol
    exporter: GraphExporterProtocol | None
    telemetry: TelemetryHooksProtocol


@runtime_checkable
class AdapterFactoryProtocol(Protocol):
    """Factory responsible for providing GraphRAG services."""

    name: str
    version: str
    supports_graphrag: bool
    dependency_hint: str

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
        """Return runtime graph services for router integration."""
        raise NotImplementedError

    def get_index_builder(self) -> GraphIndexBuilderProtocol | None:
        """Return an optional index builder for ingestion paths."""
        raise NotImplementedError

    def get_telemetry_hooks(self) -> TelemetryHooksProtocol:
        """Return telemetry hooks for router/export instrumentation."""
        raise NotImplementedError
