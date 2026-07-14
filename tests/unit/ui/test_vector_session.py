"""Vector index and client ownership tests for Streamlit session state."""

from __future__ import annotations

import gc
import weakref

import pytest

from src.ui.router_session import replace_session_router, retire_all_session_routers
from src.ui.vector_session import (
    VectorIndexResource,
    clear_stale_session_vector_resource,
    replace_session_vector_resource,
    retire_all_vector_resources,
    session_vector_resource_is_current,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_registries() -> None:
    retire_all_session_routers()
    retire_all_vector_resources()
    yield
    retire_all_session_routers()
    retire_all_vector_resources()


class _Client:
    def __init__(self, events: list[str], name: str) -> None:
        self._events = events
        self._name = name
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self._events.append(self._name)


class _Router:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._events.append("router")


def test_resource_closes_client_exactly_once() -> None:
    events: list[str] = []
    client = _Client(events, "client")
    resource = VectorIndexResource(object(), client=client)

    resource.close()
    resource.close()

    assert resource.closed
    assert client.close_calls == 1


def test_abandoned_resource_finalizer_closes_client() -> None:
    events: list[str] = []
    client = _Client(events, "client")
    resource = VectorIndexResource(object(), client=client)
    resource_ref = weakref.ref(resource)

    del resource
    gc.collect()

    assert resource_ref() is None
    assert client.close_calls == 1


def test_replacement_retires_router_before_old_client() -> None:
    events: list[str] = []
    state: dict[str, object] = {}
    old_client = _Client(events, "old-client")
    old = VectorIndexResource("old-index", client=old_client)
    new = VectorIndexResource("new-index", client=_Client(events, "new-client"))
    replace_session_vector_resource(state, old, runtime_generation=1)
    replace_session_router(state, _Router(events), runtime_generation=1)  # type: ignore[arg-type]

    replace_session_vector_resource(state, new, runtime_generation=1)

    assert events == ["router", "old-client"]
    assert state["vector_index"] == "new-index"
    assert session_vector_resource_is_current(state, runtime_generation=1)


def test_global_retirement_covers_two_sessions_and_generation_change() -> None:
    events: list[str] = []
    first_state: dict[str, object] = {}
    second_state: dict[str, object] = {}
    first = VectorIndexResource("first", client=_Client(events, "first-client"))
    second = VectorIndexResource("second", client=_Client(events, "second-client"))
    replace_session_vector_resource(first_state, first, runtime_generation=1)
    replace_session_vector_resource(second_state, second, runtime_generation=1)

    retire_all_vector_resources()

    assert events == ["first-client", "second-client"]
    assert not session_vector_resource_is_current(first_state, runtime_generation=2)
    assert not session_vector_resource_is_current(second_state, runtime_generation=2)

    clear_stale_session_vector_resource(second_state, runtime_generation=2)
    assert second_state["vector_index"] is None
