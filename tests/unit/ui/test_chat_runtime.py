"""Lifecycle tests for the process-wide Chat coordinator owner."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ui import chat_runtime
from src.ui.router_session import replace_session_router, session_router_is_current

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_runtime() -> None:
    chat_runtime.invalidate_coordinator()
    yield
    chat_runtime.invalidate_coordinator()


def test_invalidate_without_resource_does_not_construct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[object] = []
    monkeypatch.setattr(
        chat_runtime.coordinator_module,
        "MultiAgentCoordinator",
        lambda **_kwargs: constructed.append(object()),
    )

    chat_runtime.invalidate_coordinator()

    assert constructed == []


def test_same_generation_reuses_one_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Coordinator:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    monkeypatch.setattr(
        chat_runtime.coordinator_module,
        "MultiAgentCoordinator",
        _Coordinator,
    )
    store = object()

    first = chat_runtime.get_coordinator(
        cache_version=1,
        checkpointer_path=Path("chat.db"),
        store=store,
    )
    second = chat_runtime.get_coordinator(
        cache_version=1,
        checkpointer_path=Path("chat.db"),
        store=store,
    )

    assert second is first
    assert first.close_count == 0


def test_new_generation_closes_previous_coordinator_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Coordinator:
        def __init__(self, **_kwargs: object) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    monkeypatch.setattr(
        chat_runtime.coordinator_module,
        "MultiAgentCoordinator",
        _Coordinator,
    )
    store = object()
    first = chat_runtime.get_coordinator(
        cache_version=1,
        checkpointer_path=Path("chat.db"),
        store=store,
    )

    second = chat_runtime.get_coordinator(
        cache_version=2,
        checkpointer_path=Path("chat.db"),
        store=store,
    )

    assert second is not first
    assert first.close_count == 1
    chat_runtime.invalidate_coordinator()
    assert second.close_count == 1


def test_new_generation_retires_two_session_routers_before_old_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _Coordinator:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def close(self) -> None:
            events.append("coordinator")

    class _Router:
        def __init__(self, name: str) -> None:
            self.name = name
            self.closed = False

        def close(self) -> None:
            if self.closed:
                return
            self.closed = True
            events.append(self.name)

    monkeypatch.setattr(
        chat_runtime.coordinator_module,
        "MultiAgentCoordinator",
        _Coordinator,
    )
    store = object()
    chat_runtime.get_coordinator(
        cache_version=1,
        checkpointer_path=Path("chat.db"),
        store=store,
    )
    first_state: dict[str, object] = {}
    second_state: dict[str, object] = {}
    replace_session_router(
        first_state,
        _Router("router-a"),
        runtime_generation=1,
    )  # type: ignore[arg-type]
    replace_session_router(
        second_state,
        _Router("router-b"),
        runtime_generation=1,
    )  # type: ignore[arg-type]

    chat_runtime.get_coordinator(
        cache_version=2,
        checkpointer_path=Path("chat.db"),
        store=store,
    )

    assert events[:3] == ["router-a", "router-b", "coordinator"]
    assert not session_router_is_current(first_state, runtime_generation=2)
    assert not session_router_is_current(second_state, runtime_generation=2)
