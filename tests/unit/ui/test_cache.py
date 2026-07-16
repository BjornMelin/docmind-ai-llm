"""Unit tests for Streamlit cache helpers.

Employs a Streamlit stub and a minimal settings-like object to validate
cache version bumping and safe clearing behavior.
"""

from __future__ import annotations

import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

import src.ui.cache as cache_mod


class _CacheAPI:
    def __init__(self) -> None:
        self.cleared_data = 0
        self.cleared_resource = 0

    def clear(self) -> None:  # pragma: no cover - trivial
        # This method will be bound to either resource or data cache in stub
        pass


class _StreamlitStub:
    def __init__(self) -> None:
        self.cache_data = types.SimpleNamespace(clear=self._clear_data)
        self.cache_resource = types.SimpleNamespace(clear=self._clear_resource)
        self.session_state: dict[str, object] = {}
        self._api = _CacheAPI()

    def _clear_data(self) -> None:  # pragma: no cover - trivial
        self._api.cleared_data += 1

    def _clear_resource(self) -> None:  # pragma: no cover - trivial
        self._api.cleared_resource += 1


@pytest.mark.unit
def test_clear_caches_bumps_version_and_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch module-level streamlit object
    stub = _StreamlitStub()
    monkeypatch.setattr(cache_mod, "st", stub, raising=True)

    settings_obj = types.SimpleNamespace(cache_version=0)

    new_version = cache_mod.clear_caches(settings_obj)
    assert new_version == 1
    assert settings_obj.cache_version == 1

    # Ensure best-effort clears invoked
    assert stub._api.cleared_data == 1
    assert stub._api.cleared_resource == 1


@pytest.mark.unit
def test_clear_caches_closes_router_before_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.ui import chat_runtime
    from src.ui.router_session import replace_session_router
    from src.ui.vector_session import (
        VectorIndexResource,
        replace_session_vector_resource,
    )

    events: list[str] = []

    class _Router:
        def close(self) -> None:
            events.append("router")

    class _Client:
        def close(self) -> None:
            events.append("vector")

    stub = _StreamlitStub()
    replace_session_vector_resource(
        stub.session_state,
        VectorIndexResource(object(), client=_Client()),
        runtime_generation=0,
    )
    replace_session_router(
        stub.session_state,
        _Router(),
        runtime_generation=0,
    )
    monkeypatch.setattr(cache_mod, "st", stub, raising=True)
    monkeypatch.setattr(
        chat_runtime,
        "invalidate_coordinator",
        lambda: events.append("coordinator"),
    )

    cache_mod.clear_caches(types.SimpleNamespace(cache_version=0))

    assert events == ["router", "vector", "coordinator"]
    assert stub.session_state["router_engine"] is None
    assert stub.session_state["vector_index"] is None


@pytest.mark.unit
def test_clear_caches_rejects_active_job_without_mutating_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.ui.background_jobs import (
        JobConflictError,
        _reset_job_manager_for_tests,
        get_job_manager,
    )
    from src.ui.router_session import replace_session_router
    from src.ui.vector_session import (
        VectorIndexResource,
        replace_session_vector_resource,
    )

    started = threading.Event()
    release = threading.Event()
    stub = _StreamlitStub()
    resource = VectorIndexResource("index")

    class _Router:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    router = _Router()
    replace_session_vector_resource(stub.session_state, resource, runtime_generation=3)
    replace_session_router(stub.session_state, router, runtime_generation=3)
    monkeypatch.setattr(cache_mod, "st", stub, raising=True)
    settings_obj = types.SimpleNamespace(cache_version=3)
    _reset_job_manager_for_tests()
    manager = get_job_manager()

    def _work(_cancel_event, _report):  # type: ignore[no-untyped-def]
        started.set()
        assert release.wait(5)

    try:
        job_id = manager.start_job(owner_id="owner", fn=_work)
        assert started.wait(5)
        with pytest.raises(JobConflictError):
            cache_mod.clear_caches(settings_obj)

        assert settings_obj.cache_version == 3
        assert stub.session_state["vector_index"] == "index"
        assert stub.session_state["router_engine"] is router
        assert not resource.closed
        assert router.close_calls == 0
        release.set()
        assert manager.wait_for_completion(job_id, owner_id="owner") == "succeeded"
    finally:
        release.set()
        _reset_job_manager_for_tests()


@pytest.mark.unit
def test_clear_caches_fence_rejects_job_start_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.ui import vector_session
    from src.ui.background_jobs import (
        JobAdmissionPausedError,
        _reset_job_manager_for_tests,
        get_job_manager,
    )

    entered_retirement = threading.Event()
    release_retirement = threading.Event()
    stub = _StreamlitStub()
    monkeypatch.setattr(cache_mod, "st", stub, raising=True)

    def _block_retirement(*_args: object, **_kwargs: object) -> None:
        entered_retirement.set()
        assert release_retirement.wait(5)

    monkeypatch.setattr(
        vector_session,
        "clear_session_runtime",
        _block_retirement,
    )
    settings_obj = types.SimpleNamespace(cache_version=0)
    _reset_job_manager_for_tests()
    manager = get_job_manager()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(cache_mod.clear_caches, settings_obj)
            assert entered_retirement.wait(5)
            with pytest.raises(JobAdmissionPausedError):
                manager.start_job(
                    owner_id="owner",
                    fn=lambda _cancel, _report: None,
                )
            release_retirement.set()
            assert future.result(timeout=5) == 1
    finally:
        release_retirement.set()
        _reset_job_manager_for_tests()
