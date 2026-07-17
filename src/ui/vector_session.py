"""Explicit Streamlit ownership for vector indices and backing clients."""

from __future__ import annotations

import atexit
import threading
import weakref
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from src.ui.router_session import (
    _ROUTER_GENERATION_KEY,
    _RUNTIME_LIFECYCLE_LOCK,
    close_router,
    exchange_router_ownership,
    retire_all_session_routers,
)

_VECTOR_RESOURCE_KEY = "_vector_index_resource"
_VECTOR_GENERATION_KEY = "_vector_runtime_generation"
_RUNTIME_METADATA_KEYS = (
    "graphrag_index",
    "_snapshot_collections",
    "_snapshot_loaded_id",
)
_STATE_MISSING = object()
_REGISTERED_RESOURCES: weakref.WeakValueDictionary[int, VectorIndexResource] = (
    weakref.WeakValueDictionary()
)


def _close_vector_clients(client: Any | None, async_client: Any | None) -> None:
    """Best-effort close for both native Qdrant client owners."""
    try:
        from src.utils.storage import close_qdrant_clients

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
    with _RUNTIME_LIFECYCLE_LOCK:
        generation = int(runtime_generation)
        if (
            session_state.get(_VECTOR_RESOURCE_KEY) is resource
            and session_state.get(_VECTOR_GENERATION_KEY) == generation
        ):
            return
        replace_session_runtime(
            session_state,
            resource,
            None,
            runtime_generation=generation,
        )


def clear_session_runtime(session_state: Any, *, runtime_generation: int) -> None:
    """Atomically clear every session runtime owner and snapshot identity.

    Args:
        session_state: Mutable Streamlit session state.
        runtime_generation: Generation to publish for the cleared runtime.

    Returns:
        None.
    """
    replace_session_runtime(
        session_state,
        None,
        None,
        runtime_generation=runtime_generation,
        state_removals=_RUNTIME_METADATA_KEYS,
    )


def replace_session_runtime(
    session_state: Any,
    resource: VectorIndexResource | None,
    router: Any | None,
    *,
    runtime_generation: int,
    state_updates: Mapping[str, Any] | None = None,
    state_removals: Sequence[str] = (),
) -> None:
    """Atomically publish a vector/router runtime, then retire prior owners.

    Args:
        session_state: Mutable Streamlit session state.
        resource: New generation-bound vector resource, if any.
        router: New generation-bound router, if any.
        runtime_generation: Generation shared by the new runtime owners.
        state_updates: Additional session values to publish atomically.
        state_removals: Additional session keys to remove atomically.

    Returns:
        None.

    Raises:
        ValueError: If ``state_updates`` contains reserved runtime ownership keys.
    """
    with _RUNTIME_LIFECYCLE_LOCK:
        generation = int(runtime_generation)
        previous_resource = session_state.get(_VECTOR_RESOURCE_KEY, _STATE_MISSING)
        previous_router = session_state.get("router_engine", _STATE_MISSING)
        updates: dict[str, Any] = {
            _VECTOR_RESOURCE_KEY: resource,
            _VECTOR_GENERATION_KEY: generation,
            "vector_index": resource.index if resource is not None else None,
            "router_engine": router,
            _ROUTER_GENERATION_KEY: generation,
        }
        if state_updates:
            reserved = updates.keys() & state_updates.keys()
            if reserved:
                raise ValueError(
                    "Runtime state updates contain reserved ownership keys"
                )
            updates.update(state_updates)
        removals = tuple(key for key in state_removals if key not in updates)
        touched_keys = (*updates, *removals)
        previous_values = {
            key: session_state.get(key, _STATE_MISSING) for key in touched_keys
        }
        changed: list[str] = []
        try:
            for key, value in updates.items():
                changed.append(key)
                session_state[key] = value
            for key in removals:
                changed.append(key)
                session_state.pop(key, None)
        except Exception:
            for key in reversed(changed):
                _restore_session_value(session_state, key, previous_values[key])
            if (
                isinstance(resource, VectorIndexResource)
                and resource is not previous_resource
            ):
                resource.close()
            if router is not None and router is not previous_router:
                close_router(router)
            raise

        if (
            isinstance(previous_resource, VectorIndexResource)
            and previous_resource is not resource
        ):
            _REGISTERED_RESOURCES.pop(id(previous_resource), None)
        if resource is not None:
            _REGISTERED_RESOURCES[id(resource)] = resource
        exchange_router_ownership(
            None if previous_router is _STATE_MISSING else previous_router,
            router,
        )

        if previous_router is not _STATE_MISSING and previous_router is not router:
            close_router(previous_router)
        if (
            isinstance(previous_resource, VectorIndexResource)
            and previous_resource is not resource
        ):
            previous_resource.close()


def _restore_session_value(session_state: Any, key: str, value: Any) -> None:
    """Restore one session key after a failed runtime publication."""
    if value is _STATE_MISSING:
        session_state.pop(key, None)
    else:
        session_state[key] = value


def session_vector_resource_is_current(
    session_state: Any,
    *,
    runtime_generation: int,
) -> bool:
    """Return whether the session vector resource matches the runtime generation."""
    with _RUNTIME_LIFECYCLE_LOCK:
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
    with _RUNTIME_LIFECYCLE_LOCK:
        if _VECTOR_RESOURCE_KEY not in session_state:
            return
        if session_vector_resource_is_current(
            session_state,
            runtime_generation=runtime_generation,
        ):
            return
        clear_session_runtime(
            session_state,
            runtime_generation=runtime_generation,
        )


def retire_all_vector_resources() -> None:
    """Close every vector resource registered by a Streamlit session."""
    with _RUNTIME_LIFECYCLE_LOCK:
        resources = tuple(_REGISTERED_RESOURCES.values())
        _REGISTERED_RESOURCES.clear()
        for resource in resources:
            resource.close()


def retire_session_runtime_resources() -> None:
    """Retire routers before vector clients while their coordinator loop lives."""
    with _RUNTIME_LIFECYCLE_LOCK:
        retire_all_session_routers()
        retire_all_vector_resources()


atexit.register(retire_session_runtime_resources)

__all__ = [
    "VectorIndexResource",
    "clear_session_runtime",
    "clear_stale_session_vector_resource",
    "replace_session_runtime",
    "replace_session_vector_resource",
    "retire_all_vector_resources",
    "retire_session_runtime_resources",
    "session_vector_resource_is_current",
]
