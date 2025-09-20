"""LlamaIndex-backed adapter conforming to the GraphRAG adapter protocol."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import metadata
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from typing import Any

from packaging.version import InvalidVersion, Version

from src.retrieval.adapters.protocols import (
    AdapterFactoryProtocol,
    GraphIndexBuilderProtocol,
    GraphQueryArtifacts,
    GraphRetrieverProtocol,
    TelemetryHooksProtocol,
)

_INSTALL_HINT = (
    "LlamaIndex is required for GraphRAG features. Install extras via "
    "`pip install docmind_ai_llm[graphrag]` to enable knowledge graph tooling."
)
_MIN_LLAMA_INDEX_VERSION = Version("0.10.0")


class MissingLlamaIndexError(ImportError):
    """Raised when the expected LlamaIndex distribution is unavailable."""

    def __init__(self, message: str | None = None) -> None:
        """Initialise the import error with an optional override message."""
        super().__init__(message or _INSTALL_HINT)


class _NoopTelemetry(TelemetryHooksProtocol):
    """Telemetry hooks that deliberately perform no side effects."""

    def router_built(
        self,
        *,
        adapter_name: str,
        supports_graphrag: bool,
        tools: Sequence[str],
    ) -> None:
        return None

    def graph_exported(
        self,
        *,
        adapter_name: str,
        fmt: str,
        bytes_written: int,
        depth: int,
        seed_count: int,
    ) -> None:
        return None


class _PropertyGraphIndexBuilder(GraphIndexBuilderProtocol):
    """Thin wrapper delegating to ``PropertyGraphIndex`` constructors."""

    def __init__(self, cls: Any) -> None:
        self._cls = cls

    def build_from_documents(
        self,
        *,
        documents: Iterable[Any],
        storage_context: Any | None = None,
        llm: Any | None = None,
        embed_model: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        if not hasattr(self._cls, "from_documents"):
            raise MissingLlamaIndexError(
                "Installed llama-index distribution is missing "
                "PropertyGraphIndex.from_documents"
            )
        return self._cls.from_documents(
            documents=documents,
            storage_context=storage_context,
            llm=llm,
            embed_model=embed_model,
            **kwargs,
        )


@dataclass(frozen=True)
class _LlamaModules:
    knowledge_graph_retriever_cls: Any
    retriever_query_engine_cls: Any
    property_graph_index_cls: Any | None


def _resolve_version() -> str:
    """Return the installed llama-index version, enforcing minimum support."""
    try:
        version_str = metadata.version("llama-index")
    except PackageNotFoundError as exc:  # pragma: no cover - optional dependency
        raise MissingLlamaIndexError() from exc
    try:
        parsed = Version(version_str)
    except InvalidVersion:  # pragma: no cover - defensive
        return version_str
    if parsed < _MIN_LLAMA_INDEX_VERSION:
        raise MissingLlamaIndexError(
            f"llama-index>={_MIN_LLAMA_INDEX_VERSION} is required; detected "
            f"{version_str}. Upgrade via `pip install --upgrade llama-index`."
        )
    return version_str


def _load_modules() -> _LlamaModules:
    """Import the LlamaIndex modules required for GraphRAG."""
    try:
        retrievers = importlib.import_module("llama_index.core.retrievers")
        query_engine = importlib.import_module("llama_index.core.query_engine")
        property_graph = importlib.import_module(
            "llama_index.core.indices.property_graph"
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MissingLlamaIndexError() from exc

    knowledge_graph_cls = getattr(retrievers, "KnowledgeGraphRAGRetriever", None)
    if knowledge_graph_cls is None:
        raise MissingLlamaIndexError(
            "Installed llama-index distribution is missing KnowledgeGraphRAGRetriever"
        )

    retriever_query_engine_cls = getattr(query_engine, "RetrieverQueryEngine", None)
    if retriever_query_engine_cls is None:
        raise MissingLlamaIndexError(
            "Installed llama-index distribution is missing RetrieverQueryEngine"
        )

    property_graph_index_cls = getattr(property_graph, "PropertyGraphIndex", None)

    return _LlamaModules(
        knowledge_graph_retriever_cls=knowledge_graph_cls,
        retriever_query_engine_cls=retriever_query_engine_cls,
        property_graph_index_cls=property_graph_index_cls,
    )


class LlamaIndexAdapterFactory(AdapterFactoryProtocol):
    """Adapter factory exposing GraphRAG services backed by LlamaIndex."""

    def __init__(self) -> None:
        """Import LlamaIndex modules and cache metadata for reuse."""
        self.version = _resolve_version()
        self._modules = _load_modules()
        self._telemetry: TelemetryHooksProtocol = _NoopTelemetry()
        self.dependency_hint = _INSTALL_HINT
        self.name = "llama_index"
        self.supports_graphrag = True

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
        """Return retriever/query-engine artefacts bound to the adapter."""
        if property_graph_index is None:
            raise ValueError("property_graph_index is required for GraphRAG")

        storage_context = getattr(property_graph_index, "storage_context", None)
        if storage_context is None:
            raise ValueError(
                "property_graph_index must expose a storage_context for GraphRAG"
            )

        retriever: GraphRetrieverProtocol = self._modules.knowledge_graph_retriever_cls(
            storage_context=storage_context,
            verbose=False,
            graph_traversal_depth=path_depth,
            max_knowledge_sequence=similarity_top_k,
        )

        engine_kwargs: dict[str, Any] = {
            "retriever": retriever,
            "llm": llm,
            "response_mode": response_mode,
            "verbose": False,
        }
        if node_postprocessors:
            engine_kwargs["node_postprocessors"] = list(node_postprocessors)
        query_engine = self._modules.retriever_query_engine_cls.from_args(
            **engine_kwargs
        )
        telemetry = self.get_telemetry_hooks()
        return GraphQueryArtifacts(
            retriever=retriever,
            query_engine=query_engine,
            exporter=None,
            telemetry=telemetry,
        )

    def get_index_builder(self) -> GraphIndexBuilderProtocol | None:
        """Return an index builder when PropertyGraphIndex is importable."""
        cls = self._modules.property_graph_index_cls
        if cls is None:
            return None
        return _PropertyGraphIndexBuilder(cls)

    def get_telemetry_hooks(self) -> TelemetryHooksProtocol:
        """Expose telemetry hooks for router/export instrumentation."""
        return self._telemetry


def build_llama_index_factory() -> LlamaIndexAdapterFactory:
    """Construct and return the LlamaIndex adapter factory."""
    return LlamaIndexAdapterFactory()


def llama_index_available() -> bool:
    """Return ``True`` when llama_index appears importable."""
    return importlib.util.find_spec("llama_index.core") is not None


_ADAPTER_NS_CACHE: dict[str, SimpleNamespace | None] = {"adapter": None}


def set_llama_index_adapter(adapter: SimpleNamespace | None) -> None:
    """Inject a custom adapter namespace for testing purposes."""
    _ADAPTER_NS_CACHE["adapter"] = adapter


def _build_router_namespace() -> SimpleNamespace:
    modules = _load_modules()
    query_engine = importlib.import_module("llama_index.core.query_engine")
    selectors = importlib.import_module("llama_index.core.selectors")
    tools = importlib.import_module("llama_index.core.tools")

    def _maybe_selector(llm: Any | None) -> Any | None:
        factory = getattr(selectors, "PydanticSingleSelector", None)
        if factory is None:
            return None
        try:
            return factory.from_defaults(llm=llm)
        except ImportError:  # pragma: no cover - defensive
            return None

    return SimpleNamespace(
        RouterQueryEngine=query_engine.RouterQueryEngine,
        RetrieverQueryEngine=modules.retriever_query_engine_cls,
        QueryEngineTool=tools.QueryEngineTool,
        ToolMetadata=tools.ToolMetadata,
        LLMSingleSelector=selectors.LLMSingleSelector,
        get_pydantic_selector=_maybe_selector,
        __is_stub__=False,
        supports_graphrag=True,
        graphrag_disabled_reason="",
    )


def get_llama_index_adapter(force_reload: bool = False) -> SimpleNamespace:
    """Return a namespace exposing the router components from llama_index."""
    cached = _ADAPTER_NS_CACHE.get("adapter")
    if cached is not None and not force_reload:
        return cached
    if not llama_index_available():
        raise MissingLlamaIndexError()
    adapter_ns = _build_router_namespace()
    _ADAPTER_NS_CACHE["adapter"] = adapter_ns
    return adapter_ns


__all__ = [
    "LlamaIndexAdapterFactory",
    "MissingLlamaIndexError",
    "build_llama_index_factory",
    "get_llama_index_adapter",
    "llama_index_available",
    "set_llama_index_adapter",
]
