"""Router ownership tests for Streamlit session state."""

from __future__ import annotations

import pytest

from src.ui.router_session import (
    replace_session_router,
    retire_all_session_routers,
    session_router_is_current,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_router_registry() -> None:
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
