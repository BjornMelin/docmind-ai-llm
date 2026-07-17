"""Router ownership tests for Streamlit session state."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.ui.router_session import (
    replace_session_router,
    retire_all_session_routers,
    session_router_is_current,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_router_registry() -> Iterator[None]:
    retire_all_session_routers()
    yield
    retire_all_session_routers()


class _Router:
    def __init__(self, *, fail_close: bool = False) -> None:
        self.close_calls = 0
        self._fail_close = fail_close
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.close_calls += 1
        if self._fail_close:
            raise RuntimeError("cleanup failed")


class _FailingState(dict[str, object]):
    def __init__(self, initial: dict[str, object], fail_key: str) -> None:
        super().__init__(initial)
        self._fail_key = fail_key
        self._armed = True

    def __setitem__(self, key: str, value: object) -> None:
        super().__setitem__(key, value)
        if self._armed and key == self._fail_key:
            self._armed = False
            raise RuntimeError("publication failed")


class _BlockingGenerationState:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.generation_read = threading.Event()
        self.release_generation_read = threading.Event()
        self._block_next_generation_read = False

    def arm_generation_read(self) -> None:
        self._block_next_generation_read = True

    def get(self, key: str, default: object = None) -> object:
        if key == "_router_runtime_generation" and self._block_next_generation_read:
            self._block_next_generation_read = False
            self.generation_read.set()
            assert self.release_generation_read.wait(5)
        return self.values.get(key, default)

    def __setitem__(self, key: str, value: object) -> None:
        self.values[key] = value

    def pop(self, key: str, default: object = None) -> object:
        return self.values.pop(key, default)


def test_replace_closes_prior_router_once() -> None:
    old = _Router()
    new = _Router()
    state = {"router_engine": old}

    replace_session_router(state, new, runtime_generation=1)  # type: ignore[arg-type]

    assert state["router_engine"] is new
    assert old.close_calls == 1
    assert new.close_calls == 0


def test_same_router_is_not_closed() -> None:
    router = _Router()
    state = {"router_engine": router}

    replace_session_router(state, router, runtime_generation=1)  # type: ignore[arg-type]

    assert state["router_engine"] is router
    assert router.close_calls == 0


def test_cleanup_error_does_not_block_replacement() -> None:
    old = _Router(fail_close=True)
    new = _Router()
    state = {"router_engine": old}

    replace_session_router(state, new, runtime_generation=1)  # type: ignore[arg-type]

    assert state["router_engine"] is new
    assert old.close_calls == 1


def test_clear_closes_prior_router() -> None:
    old = _Router()
    state = {"router_engine": old}

    replace_session_router(state, None, runtime_generation=1)

    assert state["router_engine"] is None
    assert old.close_calls == 1


def test_generation_invalidation_retires_two_sessions() -> None:
    first = _Router()
    second = _Router()
    first_state: dict[str, object] = {}
    second_state: dict[str, object] = {}
    replace_session_router(first_state, first, runtime_generation=1)  # type: ignore[arg-type]
    replace_session_router(second_state, second, runtime_generation=1)  # type: ignore[arg-type]

    retire_all_session_routers()

    assert first.close_calls == 1
    assert second.close_calls == 1
    assert not session_router_is_current(first_state, runtime_generation=2)
    assert not session_router_is_current(second_state, runtime_generation=2)

    replacement = _Router()
    replace_session_router(second_state, replacement, runtime_generation=2)  # type: ignore[arg-type]
    assert second.close_calls == 1
    assert replacement.close_calls == 0
    assert session_router_is_current(second_state, runtime_generation=2)


def test_current_check_serializes_with_generation_replacement() -> None:
    router = _Router()
    state = _BlockingGenerationState()
    replace_session_router(state, router, runtime_generation=1)  # type: ignore[arg-type]
    state.arm_generation_read()
    replacement_started = threading.Event()
    replacement_finished = threading.Event()

    def _replace_generation() -> None:
        replacement_started.set()
        replace_session_router(state, router, runtime_generation=2)  # type: ignore[arg-type]
        replacement_finished.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            current = executor.submit(
                session_router_is_current,
                state,
                runtime_generation=1,
            )
            assert state.generation_read.wait(5)
            replacement = executor.submit(_replace_generation)
            assert replacement_started.wait(5)
            assert not replacement_finished.wait(0.1)

            state.release_generation_read.set()
            assert current.result(timeout=5)
            replacement.result(timeout=5)

        assert not session_router_is_current(state, runtime_generation=1)
        assert session_router_is_current(state, runtime_generation=2)
    finally:
        state.release_generation_read.set()


@pytest.mark.parametrize(
    "fail_key",
    ["router_engine", "_router_runtime_generation"],
)
def test_failed_router_publication_preserves_old_owner(fail_key: str) -> None:
    old = _Router()
    new = _Router()
    state = _FailingState(
        {
            "router_engine": old,
            "_router_runtime_generation": 1,
        },
        fail_key,
    )

    with pytest.raises(RuntimeError, match="publication failed"):
        replace_session_router(state, new, runtime_generation=2)  # type: ignore[arg-type]

    assert state["router_engine"] is old
    assert state["_router_runtime_generation"] == 1
    assert old.close_calls == 0
    assert new.close_calls == 1
