"""Streamlit session ownership for router query engines."""

from __future__ import annotations

import threading
import weakref
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.retrieval.router_factory import DocMindRouterQueryEngine

_MISSING = object()
_ROUTER_GENERATION_KEY = "_router_runtime_generation"
_RUNTIME_LIFECYCLE_LOCK = threading.RLock()
_REGISTERED_ROUTERS: weakref.WeakValueDictionary[int, Any] = (
    weakref.WeakValueDictionary()
)
_NON_WEAK_ROUTERS: dict[int, Any] = {}


def close_router(router: Any) -> None:
    """Close one router at the UI ownership boundary."""
    close = getattr(router, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:  # pragma: no cover - defensive cleanup boundary
        logger.debug(
            "Router cleanup failed (error_type={})",
            type(exc).__name__,
        )


def exchange_router_ownership(previous: Any, replacement: Any) -> None:
    """Update the process router registry without closing either owner."""
    with _RUNTIME_LIFECYCLE_LOCK:
        if previous is not _MISSING and previous is not replacement:
            _REGISTERED_ROUTERS.pop(id(previous), None)
            _NON_WEAK_ROUTERS.pop(id(previous), None)
        if replacement is not None:
            try:
                _REGISTERED_ROUTERS[id(replacement)] = replacement
            except TypeError:
                _NON_WEAK_ROUTERS[id(replacement)] = replacement


def replace_session_router(
    session_state: Any,
    router: DocMindRouterQueryEngine | None,
    *,
    runtime_generation: int,
) -> None:
    """Replace the generation-bound session router and close its predecessor."""
    with _RUNTIME_LIFECYCLE_LOCK:
        previous = session_state.get("router_engine", _MISSING)
        generation = int(runtime_generation)
        if (
            previous is router
            and session_state.get(_ROUTER_GENERATION_KEY) == generation
        ):
            return

        previous_generation = session_state.get(_ROUTER_GENERATION_KEY, _MISSING)
        try:
            session_state["router_engine"] = router
            session_state[_ROUTER_GENERATION_KEY] = generation
        except Exception:
            _restore_session_value(session_state, "router_engine", previous)
            _restore_session_value(
                session_state,
                _ROUTER_GENERATION_KEY,
                previous_generation,
            )
            if router is not None and router is not previous:
                close_router(router)
            raise

        exchange_router_ownership(previous, router)
        if previous is not _MISSING and previous is not router:
            close_router(previous)


def _restore_session_value(session_state: Any, key: str, value: Any) -> None:
    """Restore one session key after a failed ownership publication."""
    if value is _MISSING:
        session_state.pop(key, None)
    else:
        session_state[key] = value


def session_router_is_current(session_state: Any, *, runtime_generation: int) -> bool:
    """Return whether the session owns a live router for this runtime generation."""
    router = session_state.get("router_engine")
    if router is None or session_state.get(_ROUTER_GENERATION_KEY) != int(
        runtime_generation
    ):
        return False
    with _RUNTIME_LIFECYCLE_LOCK:
        registered = _REGISTERED_ROUTERS.get(id(router))
        if registered is None:
            registered = _NON_WEAK_ROUTERS.get(id(router))
        return registered is router


def retire_all_session_routers() -> None:
    """Close every registered session router before its owner loop is retired."""
    with _RUNTIME_LIFECYCLE_LOCK:
        routers = (*_REGISTERED_ROUTERS.values(), *_NON_WEAK_ROUTERS.values())
        _REGISTERED_ROUTERS.clear()
        _NON_WEAK_ROUTERS.clear()
        for router in routers:
            close_router(router)


__all__ = [
    "close_router",
    "exchange_router_ownership",
    "replace_session_router",
    "retire_all_session_routers",
    "session_router_is_current",
]
