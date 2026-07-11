"""Lazy LlamaIndex router imports and GraphRAG availability checks."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module, util
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

LLAMA_INDEX_INSTALL_HINT = (
    "llama-index-core is a required DocMind dependency. Reinstall DocMind with "
    "its required dependencies or provide a stub adapter for isolated tests."
)


class MissingLlamaIndexError(ImportError):
    """Raised when ``llama_index.core`` cannot be imported."""

    def __init__(self, message: str | None = None) -> None:
        """Initialise the error with an optional custom message."""
        super().__init__(message or LLAMA_INDEX_INSTALL_HINT)


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
    """Return ``True`` when the required ``llama_index.core`` package exists."""
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


@lru_cache(maxsize=1)
def _cached_graphrag_health() -> tuple[bool, str, str]:
    """Return whether required LlamaIndex PropertyGraph APIs are available."""
    try:
        module = import_module("llama_index.core.indices.property_graph")
    except ImportError:
        return False, "unavailable", LLAMA_INDEX_INSTALL_HINT
    if getattr(module, "PropertyGraphIndex", None) is None:
        return False, "unavailable", LLAMA_INDEX_INSTALL_HINT
    return True, "llama_index", "LlamaIndex core PropertyGraphIndex is available."


def get_graphrag_health(*, force_refresh: bool = False) -> tuple[bool, str, str]:
    """Return GraphRAG support, backend name, and operator guidance."""
    if force_refresh:
        _cached_graphrag_health.cache_clear()
    return _cached_graphrag_health()


__all__ = [
    "LLAMA_INDEX_INSTALL_HINT",
    "LlamaIndexAdapterProtocol",
    "MissingLlamaIndexError",
    "get_graphrag_health",
    "get_llama_index_adapter",
    "llama_index_available",
    "set_llama_index_adapter",
]
