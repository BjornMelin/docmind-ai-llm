"""Vector index and client ownership tests for Streamlit session state."""

from __future__ import annotations

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.ui.router_session import (
    replace_session_router,
    retire_all_session_routers,
    session_router_is_current,
)
from src.ui.vector_session import (
    VectorIndexResource,
    clear_session_runtime,
    clear_stale_session_vector_resource,
    replace_session_runtime,
    replace_session_vector_resource,
    retire_all_vector_resources,
    retire_session_runtime_resources,
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


class _FailingState(dict[str, object]):
    def __init__(self, initial: dict[str, object], fail_key: str) -> None:
        super().__init__(initial)
        self._fail_key = fail_key
        self._armed = True

    def __setitem__(self, key: str, value: object) -> None:
        super().__setitem__(key, value)
        if self._armed and key == self._fail_key:
            self._armed = False
            raise RuntimeError(f"publication failed for {key}")

    def pop(self, key: str, *args: object) -> object:
        value = super().pop(key, *args)
        if self._armed and key == self._fail_key:
            self._armed = False
            raise RuntimeError(f"publication failed for {key}")
        return value


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


@pytest.mark.parametrize(
    "fail_key",
    [
        "_vector_index_resource",
        "_vector_runtime_generation",
        "vector_index",
        "router_engine",
        "_router_runtime_generation",
        "_snapshot_collections",
        "graphrag_index",
        "_snapshot_loaded_id",
    ],
)
def test_runtime_publication_failure_restores_every_session_key(
    fail_key: str,
) -> None:
    events: list[str] = []
    initial_state: dict[str, object] = {}
    old_client = _Client(events, "old-client")
    old_resource = VectorIndexResource("old-index", client=old_client)
    old_router = _Router(events)
    replace_session_runtime(
        initial_state,
        old_resource,
        old_router,
        runtime_generation=1,
        state_updates={
            "_snapshot_collections": {"text": "old-text", "image": "old-image"},
            "graphrag_index": "old-graph",
            "_snapshot_loaded_id": "old-snapshot",
        },
    )
    state = _FailingState(initial_state, fail_key)
    expected = dict(initial_state)
    new_client = _Client(events, "new-client")
    new_resource = VectorIndexResource("new-index", client=new_client)
    new_router = _Router(events)

    with pytest.raises(RuntimeError, match="publication failed"):
        replace_session_runtime(
            state,
            new_resource,
            new_router,
            runtime_generation=2,
            state_updates={
                "_snapshot_collections": {
                    "text": "new-text",
                    "image": "new-image",
                },
                "graphrag_index": "new-graph",
                "_snapshot_loaded_id": "new-snapshot",
            },
        )

    assert state == expected
    assert old_client.close_calls == 0
    assert new_client.close_calls == 1
    assert events == ["new-client", "router"]


def test_runtime_publication_closes_old_owners_exactly_once() -> None:
    events: list[str] = []
    state: dict[str, object] = {}
    old_client = _Client(events, "old-client")
    new_client = _Client(events, "new-client")
    old_resource = VectorIndexResource("old-index", client=old_client)
    new_resource = VectorIndexResource("new-index", client=new_client)
    old_router = _Router(events)
    new_router = _Router(events)
    replace_session_runtime(
        state,
        old_resource,
        old_router,
        runtime_generation=1,
    )

    replace_session_runtime(
        state,
        new_resource,
        new_router,
        runtime_generation=2,
    )

    assert events == ["router", "old-client"]
    assert old_client.close_calls == 1
    assert new_client.close_calls == 0
    assert state["router_engine"] is new_router
    assert state["vector_index"] == "new-index"


def test_clear_session_runtime_removes_all_runtime_identity() -> None:
    events: list[str] = []
    state: dict[str, object] = {}
    client = _Client(events, "client")
    resource = VectorIndexResource("index", client=client)
    router = _Router(events)
    replace_session_runtime(
        state,
        resource,
        router,
        runtime_generation=1,
        state_updates={
            "graphrag_index": "graph",
            "_snapshot_collections": {"text": "text"},
            "_snapshot_loaded_id": "snapshot",
        },
    )

    clear_session_runtime(state, runtime_generation=2)

    assert state == {
        "_vector_index_resource": None,
        "_vector_runtime_generation": 2,
        "vector_index": None,
        "router_engine": None,
        "_router_runtime_generation": 2,
    }
    assert events == ["router", "client"]
    assert client.close_calls == 1
    assert not session_router_is_current(state, runtime_generation=2)
    assert not session_vector_resource_is_current(state, runtime_generation=2)


@pytest.mark.parametrize(
    "fail_key",
    [
        "_vector_index_resource",
        "_vector_runtime_generation",
        "vector_index",
        "router_engine",
        "_router_runtime_generation",
        "graphrag_index",
        "_snapshot_collections",
        "_snapshot_loaded_id",
    ],
)
def test_clear_session_runtime_failure_restores_every_session_key(
    fail_key: str,
) -> None:
    events: list[str] = []
    initial_state: dict[str, object] = {}
    client = _Client(events, "client")
    resource = VectorIndexResource("index", client=client)
    router = _Router(events)
    replace_session_runtime(
        initial_state,
        resource,
        router,
        runtime_generation=1,
        state_updates={
            "graphrag_index": "graph",
            "_snapshot_collections": {"text": "text"},
            "_snapshot_loaded_id": "snapshot",
        },
    )
    state = _FailingState(initial_state, fail_key)
    expected = dict(initial_state)

    with pytest.raises(RuntimeError, match="publication failed"):
        clear_session_runtime(state, runtime_generation=2)

    assert state == expected
    assert client.close_calls == 0
    assert events == []
    assert session_router_is_current(state, runtime_generation=1)
    assert session_vector_resource_is_current(state, runtime_generation=1)


@pytest.mark.parametrize("publication_first", [True, False])
def test_publication_and_global_retirement_share_one_lifecycle_lock(
    publication_first: bool,
) -> None:
    state: dict[str, object] = {}
    close_entered = threading.Event()
    release_close = threading.Event()

    class _BlockingRouter:
        def __init__(self, *, block: bool) -> None:
            self.close_calls = 0
            self._block = block

        def close(self) -> None:
            if self.close_calls:
                return
            self.close_calls += 1
            if self._block:
                close_entered.set()
                assert release_close.wait(5)

    old_client = _Client([], "old-client")
    new_client = _Client([], "new-client")
    old_resource = VectorIndexResource("old-index", client=old_client)
    new_resource = VectorIndexResource("new-index", client=new_client)
    old_router = _BlockingRouter(block=True)
    new_router = _BlockingRouter(block=False)
    replace_session_runtime(
        state,
        old_resource,
        old_router,
        runtime_generation=1,
    )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            if publication_first:
                first = executor.submit(
                    replace_session_runtime,
                    state,
                    new_resource,
                    new_router,
                    runtime_generation=2,
                )
                second_fn = retire_session_runtime_resources
            else:
                first = executor.submit(retire_session_runtime_resources)

                def second_fn() -> None:
                    replace_session_runtime(
                        state,
                        new_resource,
                        new_router,
                        runtime_generation=2,
                    )

            assert close_entered.wait(5)
            second = executor.submit(second_fn)
            assert not second.done()
            release_close.set()
            first.result(timeout=5)
            second.result(timeout=5)

        assert old_router.close_calls == 1
        assert old_client.close_calls == 1
        if publication_first:
            assert new_router.close_calls == 1
            assert new_client.close_calls == 1
            assert not session_router_is_current(state, runtime_generation=2)
            assert not session_vector_resource_is_current(
                state,
                runtime_generation=2,
            )
        else:
            assert new_router.close_calls == 0
            assert new_client.close_calls == 0
            assert session_router_is_current(state, runtime_generation=2)
            assert session_vector_resource_is_current(
                state,
                runtime_generation=2,
            )
    finally:
        release_close.set()
