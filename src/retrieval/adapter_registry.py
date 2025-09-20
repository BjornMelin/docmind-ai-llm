"""Registry for GraphRAG adapter factories."""

from __future__ import annotations

from loguru import logger

from src.retrieval.adapters.protocols import AdapterFactoryProtocol
from src.retrieval.llama_index_adapter import (
    MissingLlamaIndexError,
    build_llama_index_factory,
)


class MissingGraphAdapterError(RuntimeError):
    """Raised when no adapter is registered for GraphRAG operations."""


_REGISTRY: dict[str, AdapterFactoryProtocol] = {}


def register_adapter(factory: AdapterFactoryProtocol) -> None:
    """Register an adapter factory by name."""
    _REGISTRY[factory.name] = factory
    logger.debug("Registered graph adapter '%s'", factory.name)


def unregister_adapter(name: str) -> None:
    """Remove an adapter factory from the registry."""
    _REGISTRY.pop(name, None)


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
        logger.debug("Skipping LlamaIndex adapter registration: %s", exc)
        return
    register_adapter(factory)


# Register default adapter eagerly so callers can rely on lookups.
ensure_default_adapter()


__all__ = [
    "MissingGraphAdapterError",
    "ensure_default_adapter",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "unregister_adapter",
]
