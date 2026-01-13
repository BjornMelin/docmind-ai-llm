"""Registry for GraphRAG adapter factories."""

from __future__ import annotations

from functools import lru_cache

from loguru import logger

from src.retrieval.adapters.protocols import AdapterFactoryProtocol
from src.retrieval.llama_index_adapter import (
    MissingLlamaIndexError,
    build_llama_index_factory,
)


class MissingGraphAdapterError(RuntimeError):
    """Raised when no adapter is registered for GraphRAG operations."""


_REGISTRY: dict[str, AdapterFactoryProtocol] = {}

GRAPH_DEPENDENCY_HINT = (
    "GraphRAG disabled: install optional extras 'docmind_ai_llm[graph]' "
    "to enable knowledge graph retrieval."
)


def register_adapter(factory: AdapterFactoryProtocol) -> None:
    """Register an adapter factory by name."""
    _REGISTRY[factory.name] = factory
    logger.debug("Registered graph adapter '{}'", factory.name)
    _cached_default_adapter_health.cache_clear()


def unregister_adapter(name: str) -> None:
    """Remove an adapter factory from the registry."""
    _REGISTRY.pop(name, None)
    _cached_default_adapter_health.cache_clear()


def get_adapter(name: str | None = None) -> AdapterFactoryProtocol:
    """Return a registered adapter, preferring LlamaIndex by default."""
    if name:
        try:
            return _REGISTRY[name]
        except KeyError as exc:
            raise MissingGraphAdapterError(
                f"No graph adapter registered under '{name}'"
            ) from exc
    if "llama_index" in _REGISTRY:
        return _REGISTRY["llama_index"]
    if _REGISTRY:
        return next(iter(_REGISTRY.values()))
    raise MissingGraphAdapterError("No graph adapters are registered")


def list_adapters() -> list[str]:
    """Return the list of registered adapter names."""
    return sorted(_REGISTRY)


def ensure_default_adapter() -> None:
    """Ensure the default LlamaIndex adapter is registered when available."""
    if "llama_index" in _REGISTRY:
        return
    try:
        factory = build_llama_index_factory()
    except MissingLlamaIndexError as exc:
        logger.debug("Skipping LlamaIndex adapter registration: {}", exc)
        return
    register_adapter(factory)


def resolve_adapter(
    adapter: AdapterFactoryProtocol | None,
) -> AdapterFactoryProtocol:
    """Return the provided adapter, or resolve the default registry adapter."""
    if adapter is not None:
        return adapter
    ensure_default_adapter()
    return get_adapter()


# Register default adapter lazily on first lookup —
# `ensure_default_adapter()` is intentionally not called here.
# ensure_default_adapter()


@lru_cache(maxsize=1)
def _cached_default_adapter_health() -> tuple[bool, str, str]:
    """Return cached GraphRAG adapter health details."""
    ensure_default_adapter()
    try:
        adapter = get_adapter()
    except MissingGraphAdapterError:
        return False, "unavailable", GRAPH_DEPENDENCY_HINT
    supports = bool(getattr(adapter, "supports_graphrag", False))
    name = getattr(adapter, "name", "unknown")
    hint = getattr(adapter, "dependency_hint", GRAPH_DEPENDENCY_HINT)
    return supports, name, hint


def get_default_adapter_health(*, force_refresh: bool = False) -> tuple[bool, str, str]:
    """Return a tuple of (supported, adapter_name, guidance)."""
    if force_refresh:
        _cached_default_adapter_health.cache_clear()
    return _cached_default_adapter_health()


__all__ = [
    "GRAPH_DEPENDENCY_HINT",
    "MissingGraphAdapterError",
    "ensure_default_adapter",
    "get_adapter",
    "get_default_adapter_health",
    "list_adapters",
    "register_adapter",
    "resolve_adapter",
    "unregister_adapter",
]
