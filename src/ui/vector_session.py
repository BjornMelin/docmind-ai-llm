"""Explicit Streamlit ownership for vector indices and backing clients."""

from __future__ import annotations

import atexit
import threading
import weakref
from typing import Any

from loguru import logger

from src.ui.router_session import replace_session_router, retire_all_session_routers
from src.utils.storage import close_qdrant_clients

_VECTOR_RESOURCE_KEY = "_vector_index_resource"
_VECTOR_GENERATION_KEY = "_vector_runtime_generation"
_REGISTRY_LOCK = threading.RLock()
_REGISTERED_RESOURCES: weakref.WeakValueDictionary[int, VectorIndexResource] = (
    weakref.WeakValueDictionary()
)


def _close_vector_clients(client: Any | None, async_client: Any | None) -> None:
    """Best-effort close for both native Qdrant client owners."""
    try:
        close_qdrant_clients(client, async_client)
    except Exception as exc:  # pragma: no cover - defensive cleanup boundary
        logger.debug(
            "Vector client cleanup failed (error_type={})",
            type(exc).__name__,
        )


class VectorIndexResource:
    """Own one vector index and the client that keeps it queryable."""

    def __init__(
        self,
        index: Any,
        *,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        """Create an index resource with its closeable backing clients."""
        self.index = index
        self._lock = threading.Lock()
        self._closed = False
        self._finalizer = weakref.finalize(
            self,
            _close_vector_clients,
            client,
            async_client,
        )

    @classmethod
    def from_vector_store(cls, index: Any, vector_store: Any) -> VectorIndexResource:
        """Own ``index`` and the client exposed by its Qdrant vector store."""
        return cls(
            index,
            client=getattr(vector_store, "client", None),
            async_client=getattr(vector_store, "_aclient", None),
        )

    @property
    def closed(self) -> bool:
        """Return whether this resource has already released its client."""
        with self._lock:
            return self._closed

    def close(self) -> None:
        """Close the backing client exactly once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._finalizer()


def replace_session_vector_resource(
    session_state: Any,
    resource: VectorIndexResource | None,
    *,
    runtime_generation: int,
) -> None:
    """Retire the router, then replace and close the prior vector resource."""
    generation = int(runtime_generation)
    previous = session_state.get(_VECTOR_RESOURCE_KEY)
    if previous is resource and session_state.get(_VECTOR_GENERATION_KEY) == generation:
        return

    # Routers can still be using the old client, so they must retire first.
    replace_session_router(
        session_state,
        None,
        runtime_generation=generation,
    )

    with _REGISTRY_LOCK:
        if isinstance(previous, VectorIndexResource) and previous is not resource:
            _REGISTERED_RESOURCES.pop(id(previous), None)
        if resource is not None:
            _REGISTERED_RESOURCES[id(resource)] = resource
        session_state[_VECTOR_RESOURCE_KEY] = resource
        session_state[_VECTOR_GENERATION_KEY] = generation
        session_state["vector_index"] = resource.index if resource is not None else None

    if isinstance(previous, VectorIndexResource) and previous is not resource:
        previous.close()


def session_vector_resource_is_current(
    session_state: Any,
    *,
    runtime_generation: int,
) -> bool:
    """Return whether the session vector resource matches the runtime generation."""
    resource = session_state.get(_VECTOR_RESOURCE_KEY)
    return (
        isinstance(resource, VectorIndexResource)
        and not resource.closed
        and session_state.get(_VECTOR_GENERATION_KEY) == int(runtime_generation)
        and session_state.get("vector_index") is resource.index
    )


def clear_stale_session_vector_resource(
    session_state: Any,
    *,
    runtime_generation: int,
) -> None:
    """Clear a session resource that belongs to an older runtime generation."""
    if _VECTOR_RESOURCE_KEY not in session_state:
        return
    if session_vector_resource_is_current(
        session_state,
        runtime_generation=runtime_generation,
    ):
        return
    replace_session_vector_resource(
        session_state,
        None,
        runtime_generation=runtime_generation,
    )


def retire_all_vector_resources() -> None:
    """Close every vector resource registered by a Streamlit session."""
    with _REGISTRY_LOCK:
        resources = tuple(_REGISTERED_RESOURCES.values())
        _REGISTERED_RESOURCES.clear()
    for resource in resources:
        resource.close()


def retire_session_runtime_resources() -> None:
    """Retire routers before vector clients while their coordinator loop lives."""
    retire_all_session_routers()
    retire_all_vector_resources()


atexit.register(retire_session_runtime_resources)

__all__ = [
    "VectorIndexResource",
    "clear_stale_session_vector_resource",
    "replace_session_vector_resource",
    "retire_all_vector_resources",
    "retire_session_runtime_resources",
    "session_vector_resource_is_current",
]
