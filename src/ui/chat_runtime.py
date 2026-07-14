"""Process-wide owner for the active Chat coordinator."""

from __future__ import annotations

import atexit
import threading
from pathlib import Path
from typing import Any

from src.agents import coordinator as coordinator_module
from src.ui.vector_session import retire_session_runtime_resources

_LOCK = threading.RLock()
_COORDINATOR: coordinator_module.MultiAgentCoordinator | None = None
_RESOURCE_KEY: tuple[int, Path, int] | None = None


def _retire_session_resources() -> None:
    """Retire loop-bound routers and their vector clients before coordinator close."""
    retire_session_runtime_resources()


def get_coordinator(
    *,
    cache_version: int,
    checkpointer_path: Path,
    store: Any,
) -> coordinator_module.MultiAgentCoordinator:
    """Return the coordinator for the current runtime generation.

    A settings generation change replaces and closes the previous coordinator,
    so provider, model, tool, and timeout changes cannot leave Chat on a stale
    graph.
    """
    global _COORDINATOR, _RESOURCE_KEY

    key = (int(cache_version), Path(checkpointer_path), id(store))
    previous: coordinator_module.MultiAgentCoordinator | None
    with _LOCK:
        if _COORDINATOR is not None and key == _RESOURCE_KEY:
            return _COORDINATOR

        replacement = coordinator_module.MultiAgentCoordinator(
            checkpointer_path=checkpointer_path,
            store=store,
        )
        previous = _COORDINATOR
        if previous is not None:
            # Keep replacement publication behind the runtime lock so another
            # session cannot register a new-generation router during retirement.
            _retire_session_resources()
        _COORDINATOR = replacement
        _RESOURCE_KEY = key
    # Coordinator close can wait for its bounded graph-runner grace. Session
    # resource retirement is fenced above; release the lock before this wait.
    if previous is not None:
        previous.close()
    return replacement


def invalidate_coordinator() -> None:
    """Detach and close the active coordinator without constructing one."""
    global _COORDINATOR, _RESOURCE_KEY

    with _LOCK:
        coordinator = _COORDINATOR
        _retire_session_resources()
        _COORDINATOR = None
        _RESOURCE_KEY = None
    if coordinator is not None:
        coordinator.close()


atexit.register(invalidate_coordinator)

__all__ = ["get_coordinator", "invalidate_coordinator"]
