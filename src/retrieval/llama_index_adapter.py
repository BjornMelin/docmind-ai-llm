"""Adapters for interacting with ``llama_index.core`` in a dependency-light way.

The real library is optional during unit tests. This module exposes helpers to
lazily import the genuine classes when available and allows tests to inject
fakes that satisfy the same surface area.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

_INSTALL_HINT = (
    "llama_index.core is required for this feature. Install it via "
    "`pip install docmind_ai_llm[llama]` or provide a stub adapter."
)


class MissingLlamaIndexError(ImportError):
    """Raised when ``llama_index.core`` cannot be imported."""

    def __init__(self, message: str | None = None) -> None:
        """Initialise the error with an optional custom message."""
        super().__init__(message or _INSTALL_HINT)


@runtime_checkable
class LlamaIndexAdapterProtocol(Protocol):
    """Protocol describing the minimal surface used by the router factory."""

    RouterQueryEngine: Any
    RetrieverQueryEngine: Any
    QueryEngineTool: Any
    ToolMetadata: Any
    LLMSingleSelector: Any
    __is_stub__: bool

    def get_pydantic_selector(self, llm: Any | None) -> Any | None:
        """Return a ``PydanticSingleSelector`` built for ``llm`` when available."""


def llama_index_available() -> bool:
    """Return ``True`` when the optional ``llama_index.core`` package exists."""
    return importlib.util.find_spec("llama_index.core") is not None


_ADAPTER_STATE: dict[str, LlamaIndexAdapterProtocol | None] = {"adapter": None}


def _get_cached_adapter() -> LlamaIndexAdapterProtocol | None:
    """Return the cached adapter if present."""
    return _ADAPTER_STATE.get("adapter")


def _set_cached_adapter(adapter: LlamaIndexAdapterProtocol | None) -> None:
    """Update the cached adapter reference."""
    _ADAPTER_STATE["adapter"] = adapter


def _build_real_adapter() -> LlamaIndexAdapterProtocol:
    """Import ``llama_index`` modules and construct an adapter namespace."""
    if not llama_index_available():
        raise MissingLlamaIndexError()
    query_engine = importlib.import_module("llama_index.core.query_engine")
    selectors = importlib.import_module("llama_index.core.selectors")
    tools = importlib.import_module("llama_index.core.tools")

    def _maybe_pydantic_selector(llm: Any | None) -> Any | None:
        factory = getattr(selectors, "PydanticSingleSelector", None)
        if factory is None:
            return None
        return factory.from_defaults(llm=llm)

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
    """Return the cached real adapter, importing it lazily when needed.

    Args:
        force_reload: When ``True``, bypass the cache and re-import modules.

    Returns:
        Adapter exposing the narrow llama_index surface consumed by routers.

    Raises:
        MissingLlamaIndexError: If llama_index.core is not installed.
    """
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


__all__ = [
    "LlamaIndexAdapterProtocol",
    "MissingLlamaIndexError",
    "get_llama_index_adapter",
    "llama_index_available",
    "set_llama_index_adapter",
]
