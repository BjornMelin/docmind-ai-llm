"""Adapters for interacting with LlamaIndex with minimal hard dependencies.

This module serves two distinct callers:

1) Router factory: needs a small surface area from ``llama_index.core`` to build
   RouterQueryEngine instances without importing LlamaIndex at module import
   time (tests can inject stubs).
2) GraphRAG adapter registry: provides an ``AdapterFactoryProtocol`` for
   GraphRAG capabilities when LlamaIndex is installed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module, metadata, util
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

from packaging.version import InvalidVersion, Version

from src.retrieval.adapters.protocols import (
    AdapterFactoryProtocol,
    GraphIndexBuilderProtocol,
    GraphQueryArtifacts,
    GraphRetrieverProtocol,
    TelemetryHooksProtocol,
)

_LLAMA_INSTALL_HINT = (
    "llama_index.core is required for this feature. Install it via "
    "`pip install docmind_ai_llm[llama]` or provide a stub adapter."
)

_GRAPH_INSTALL_HINT = (
    "GraphRAG is disabled: install optional extras 'docmind_ai_llm[graph]' "
    "to enable knowledge graph retrieval."
)

_MIN_LLAMA_INDEX_VERSION = Version("0.13.0")


class MissingLlamaIndexError(ImportError):
    """Raised when ``llama_index.core`` cannot be imported."""

    def __init__(self, message: str | None = None) -> None:
        """Initialise the error with an optional custom message."""
        super().__init__(message or _LLAMA_INSTALL_HINT)


@runtime_checkable
class LlamaIndexAdapterProtocol(Protocol):
    """Minimal LlamaIndex surface area used by the router factory."""

    RouterQueryEngine: Any
    RetrieverQueryEngine: Any
    QueryEngineTool: Any
    ToolMetadata: Any
    LLMSingleSelector: Any
    __is_stub__: bool

    def get_pydantic_selector(self, llm: Any | None) -> Any | None:
        """Return a Pydantic selector for ``llm`` when available."""


def llama_index_available() -> bool:
    """Return ``True`` when the optional ``llama_index.core`` package exists."""
    # `importlib.util.find_spec()` can raise `ValueError` when a test injects a
    # stub module into `sys.modules` with `__spec__ = None` (a common pattern
    # when mocking optional dependencies). Treat that as "not available".
    try:
        return util.find_spec("llama_index.core") is not None
    except (ImportError, ValueError):
        return False


_ROUTER_ADAPTER_STATE: dict[str, LlamaIndexAdapterProtocol | None] = {"adapter": None}


def _get_cached_adapter() -> LlamaIndexAdapterProtocol | None:
    return _ROUTER_ADAPTER_STATE.get("adapter")


def _set_cached_adapter(adapter: LlamaIndexAdapterProtocol | None) -> None:
    _ROUTER_ADAPTER_STATE["adapter"] = adapter


def _build_real_adapter() -> LlamaIndexAdapterProtocol:
    """Import LlamaIndex modules and construct an adapter namespace."""
    if not llama_index_available():
        raise MissingLlamaIndexError()

    query_engine = import_module("llama_index.core.query_engine")
    selectors = import_module("llama_index.core.selectors")
    tools = import_module("llama_index.core.tools")

    def _maybe_pydantic_selector(llm: Any | None) -> Any | None:
        factory = getattr(selectors, "PydanticSingleSelector", None)
        if factory is None:
            return None
        metadata_obj = getattr(llm, "metadata", None)
        if metadata_obj is not None and not getattr(
            metadata_obj, "is_function_calling_model", True
        ):
            return None
        try:
            return factory.from_defaults(llm=llm)
        except (ImportError, ValueError):  # pragma: no cover - defensive
            return None

    adapter = SimpleNamespace(
        RouterQueryEngine=query_engine.RouterQueryEngine,
        RetrieverQueryEngine=query_engine.RetrieverQueryEngine,
        QueryEngineTool=tools.QueryEngineTool,
        ToolMetadata=tools.ToolMetadata,
        LLMSingleSelector=selectors.LLMSingleSelector,
        get_pydantic_selector=_maybe_pydantic_selector,
        __is_stub__=False,
    )
    return adapter  # type: ignore[return-value]


def get_llama_index_adapter(force_reload: bool = False) -> LlamaIndexAdapterProtocol:
    """Return a cached adapter, importing LlamaIndex lazily when needed."""
    cached = _get_cached_adapter()
    if not force_reload and cached is not None:
        return cached
    try:
        adapter = _build_real_adapter()
    except ImportError as exc:
        raise MissingLlamaIndexError() from exc
    _set_cached_adapter(adapter)
    return adapter


def set_llama_index_adapter(adapter: LlamaIndexAdapterProtocol | None) -> None:
    """Inject a custom adapter (typically a stub) for testing.

    Passing ``None`` clears the cache.
    """
    _set_cached_adapter(adapter)


def _resolve_llama_index_version() -> str:
    """Return the installed llama-index version string when available."""
    for dist in ("llama-index", "llama-index-core"):
        try:
            raw = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
        try:
            parsed = Version(raw)
        except InvalidVersion:  # pragma: no cover - defensive
            return raw
        if parsed < _MIN_LLAMA_INDEX_VERSION:
            raise MissingLlamaIndexError(
                f"{dist}>={_MIN_LLAMA_INDEX_VERSION} is required; detected {raw}. "
                "Upgrade your dependency set."
            )
        return raw
    raise MissingLlamaIndexError(_GRAPH_INSTALL_HINT)


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
    """Thin wrapper delegating to a ``PropertyGraphIndex`` constructor."""

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
            raise MissingLlamaIndexError(_GRAPH_INSTALL_HINT)
        return self._cls.from_documents(
            documents=documents,
            storage_context=storage_context,
            llm=llm,
            embed_model=embed_model,
            **kwargs,
        )


@dataclass(frozen=True)
class _GraphModules:
    property_graph_index_cls: Any | None


def _load_graph_modules() -> _GraphModules:
    """Import modules needed to build graph indices when available."""
    try:
        property_graph = import_module("llama_index.core.indices.property_graph")
    except ImportError:  # pragma: no cover - optional dependency path
        return _GraphModules(property_graph_index_cls=None)
    return _GraphModules(
        property_graph_index_cls=getattr(property_graph, "PropertyGraphIndex", None)
    )


class LlamaIndexAdapterFactory(AdapterFactoryProtocol):
    """GraphRAG adapter factory backed by documented LlamaIndex APIs."""

    def __init__(self) -> None:
        """Initialise the factory and cache optional module lookups."""
        self.version = _resolve_llama_index_version()
        self.name = "llama_index"
        self.supports_graphrag = True
        self.dependency_hint = _GRAPH_INSTALL_HINT
        self._telemetry: TelemetryHooksProtocol = _NoopTelemetry()
        self._modules_cache: _GraphModules | None = None

    @property
    def _modules(self) -> _GraphModules:
        if self._modules_cache is None:
            self._modules_cache = _load_graph_modules()
        return self._modules_cache

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
        if property_graph_index is None or not hasattr(
            property_graph_index, "as_retriever"
        ):
            raise ValueError("property_graph_index must expose as_retriever()")

        retriever: GraphRetrieverProtocol = property_graph_index.as_retriever(
            include_text=include_text,
            similarity_top_k=similarity_top_k,
            path_depth=path_depth,
        )

        adapter = get_llama_index_adapter()
        engine_kwargs: dict[str, Any] = {
            "retriever": retriever,
            "llm": llm,
            "response_mode": response_mode,
            "verbose": False,
        }
        if node_postprocessors:
            engine_kwargs["node_postprocessors"] = list(node_postprocessors)
        query_engine = adapter.RetrieverQueryEngine.from_args(**engine_kwargs)

        telemetry = self.get_telemetry_hooks()
        return GraphQueryArtifacts(
            retriever=retriever,
            query_engine=query_engine,
            exporter=None,
            telemetry=telemetry,
        )

    def get_index_builder(self) -> GraphIndexBuilderProtocol | None:
        """Return a PropertyGraphIndex builder when available."""
        cls = self._modules.property_graph_index_cls
        if cls is None:
            return None
        return _PropertyGraphIndexBuilder(cls)

    def get_telemetry_hooks(self) -> TelemetryHooksProtocol:
        """Return telemetry hook implementation (no-op by default)."""
        return self._telemetry


def build_llama_index_factory() -> LlamaIndexAdapterFactory:
    """Return the default LlamaIndex GraphRAG factory when installed."""
    if not llama_index_available():
        raise MissingLlamaIndexError(_GRAPH_INSTALL_HINT)
    return LlamaIndexAdapterFactory()


__all__ = [
    "LlamaIndexAdapterFactory",
    "LlamaIndexAdapterProtocol",
    "MissingLlamaIndexError",
    "build_llama_index_factory",
    "get_llama_index_adapter",
    "llama_index_available",
    "set_llama_index_adapter",
]
