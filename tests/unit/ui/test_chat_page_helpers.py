"""Unit tests for Chat page helpers (01_chat.py)."""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from src.ui.router_session import replace_session_router

_PHYSICAL_COLLECTIONS = {
    "text": "physical-text-v2",
    "image": "physical-image-v2",
}


class _AppRerunRequestedError(RuntimeError):
    """Model Streamlit's rerun as immediate control-flow interruption."""


def _interrupt_on_app_rerun(
    monkeypatch: pytest.MonkeyPatch,
    captures: dict[str, list[str] | int],
) -> None:
    import streamlit as st  # type: ignore

    def _rerun(*, scope: str | None = None) -> None:
        captures["reruns"] = cast(int, captures["reruns"]) + 1
        cast(list[str], captures["rerun_scopes"]).append(scope or "default")
        raise _AppRerunRequestedError

    monkeypatch.setattr(st, "rerun", _rerun, raising=False)


def _snapshot_manifest() -> dict[str, object]:
    return {"collections": dict(_PHYSICAL_COLLECTIONS)}


@pytest.fixture(autouse=True)
def clean_streamlit_session(monkeypatch):
    # Ensure streamlit import has expected attributes for testing
    import streamlit as st  # type: ignore

    # Reset session_state and capture captions per test
    st.session_state.clear()
    captures: dict[str, list[str] | int] = {
        "captions": [],
        "infos": [],
        "warnings": [],
        "successes": [],
        "errors": [],
        "markdown": [],
        "images": [],
        "chat_roles": [],
        "reruns": 0,
        "rerun_scopes": [],
    }

    def _chat_message(role: str):  # type: ignore[no-untyped-def]
        captures["chat_roles"].append(str(role))  # type: ignore[index]
        return contextlib.nullcontext()

    monkeypatch.setattr(
        st,
        "caption",
        lambda msg: captures["captions"].append(str(msg)),  # type: ignore[index]
        raising=False,
    )
    monkeypatch.setattr(
        st,
        "info",
        lambda msg: captures["infos"].append(str(msg)),  # type: ignore[index]
        raising=False,
    )
    monkeypatch.setattr(
        st,
        "warning",
        lambda msg: captures["warnings"].append(str(msg)),  # type: ignore[index]
        raising=False,
    )
    monkeypatch.setattr(
        st,
        "success",
        lambda msg: captures["successes"].append(str(msg)),  # type: ignore[index]
        raising=False,
    )
    monkeypatch.setattr(
        st,
        "error",
        lambda msg: captures["errors"].append(str(msg)),  # type: ignore[index]
        raising=False,
    )
    monkeypatch.setattr(
        st,
        "markdown",
        lambda msg: captures["markdown"].append(str(msg)),  # type: ignore[index]
        raising=False,
    )
    monkeypatch.setattr(
        st,
        "image",
        lambda img, **_k: captures["images"].append(str(img)),  # type: ignore[index]
        raising=False,
    )
    monkeypatch.setattr(
        st, "expander", lambda *_a, **_k: contextlib.nullcontext(), raising=False
    )
    monkeypatch.setattr(st, "sidebar", contextlib.nullcontext(), raising=False)
    monkeypatch.setattr(st, "subheader", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "progress", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "slider", lambda *_a, **_k: 1, raising=False)
    monkeypatch.setattr(
        st,
        "columns",
        lambda n: [SimpleNamespace() for _ in range(int(n))],
        raising=False,
    )
    monkeypatch.setattr(st, "chat_message", _chat_message, raising=False)
    monkeypatch.setattr(st, "chat_input", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(
        st,
        "spinner",
        lambda *_a, **_k: contextlib.nullcontext(),
        raising=False,
    )
    monkeypatch.setattr(st, "divider", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "write", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "selectbox", lambda *_a, **_k: "session", raising=False)
    monkeypatch.setattr(st, "text_input", lambda *_a, **_k: "", raising=False)
    monkeypatch.setattr(st, "button", lambda *_a, **_k: False, raising=False)
    monkeypatch.setattr(st, "checkbox", lambda *_a, **_k: False, raising=False)
    monkeypatch.setattr(st, "file_uploader", lambda *_a, **_k: None, raising=False)

    def _rerun(*, scope: str | None = None) -> None:
        captures["reruns"] = cast(int, captures["reruns"]) + 1
        cast(list[str], captures["rerun_scopes"]).append(scope or "default")

    monkeypatch.setattr(st, "rerun", _rerun, raising=False)
    return captures


@pytest.mark.unit
def test_get_settings_override_forwarding(monkeypatch):
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    import streamlit as st  # type: ignore

    st.session_state.clear()
    assert page._get_settings_override() is None

    replace_session_router(
        st.session_state,
        object(),
        runtime_generation=page.settings.cache_version,
    )
    ov = page._get_settings_override()
    assert ov == {"router_engine": st.session_state["router_engine"]}

    st.session_state["vector_index"] = object()
    st.session_state["graphrag_index"] = object()
    ov = page._get_settings_override()
    assert ov == {"router_engine": st.session_state["router_engine"]}


@pytest.mark.unit
def test_hydrate_router_from_snapshot(monkeypatch):
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    import streamlit as st  # type: ignore

    # Patch loader helpers to deterministic stubs
    vector_store = SimpleNamespace(client=SimpleNamespace(close=lambda: None))
    vector_index = SimpleNamespace(vector_store=vector_store)
    monkeypatch.setattr(page, "load_manifest", lambda _p: _snapshot_manifest())
    monkeypatch.setattr(page, "load_vector_index", lambda _p: vector_index)
    monkeypatch.setattr(page, "load_property_graph_index", lambda _p: "KG")
    router_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _build_router(*args, **kwargs):  # type: ignore[no-untyped-def]
        router_calls.append((args, kwargs))
        return "ROUTER"

    monkeypatch.setattr(page, "build_router_engine", _build_router)

    st.session_state.clear()
    page._hydrate_router_from_snapshot(Path("/tmp/snap"))
    assert st.session_state["vector_index"] is vector_index
    assert st.session_state["graphrag_index"] == "KG"
    assert st.session_state["router_engine"] == "ROUTER"
    assert router_calls == [
        (
            (vector_index, "KG", page.settings),
            {
                "text_collection": "physical-text-v2",
                "image_collection": "physical-image-v2",
            },
        )
    ]
    assert st.session_state["_snapshot_collections"] == _PHYSICAL_COLLECTIONS

    class _OldRouter:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    old = _OldRouter()
    # Error path: build_router_engine raises → clear the partial runtime.
    monkeypatch.setattr(
        page,
        "build_router_engine",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("x")),
    )
    st.session_state.clear()
    st.session_state["router_engine"] = old
    with pytest.raises(
        RuntimeError, match="Activated snapshot router construction failed"
    ):
        page._hydrate_router_from_snapshot(Path("/tmp/snap"))
    assert st.session_state.get("router_engine") is None
    assert old.closed == 1
    assert st.session_state["vector_index"] is None
    assert "graphrag_index" not in st.session_state


@pytest.mark.unit
def test_hydrate_required_graph_failure_clears_runtime(monkeypatch):
    """A graph-declared snapshot cannot silently activate vector-only."""
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")
    close_calls = {"sync": 0, "async": 0}

    class _SyncClient:
        def close(self) -> None:
            close_calls["sync"] += 1

    class _AsyncClient:
        async def close(self) -> None:
            close_calls["async"] += 1

    vector_store = SimpleNamespace(client=_SyncClient(), _aclient=_AsyncClient())
    vector_index = SimpleNamespace(vector_store=vector_store)
    manifest = _snapshot_manifest()
    manifest["graph_store_type"] = "property_graph"
    monkeypatch.setattr(page, "load_manifest", lambda _p: manifest)
    monkeypatch.setattr(page, "load_vector_index", lambda _p: vector_index)
    monkeypatch.setattr(page, "load_property_graph_index", lambda _p: None)

    st.session_state["router_engine"] = object()
    with pytest.raises(RuntimeError, match="property graph is unavailable"):
        page._hydrate_router_from_snapshot(Path("sid"))

    assert st.session_state.get("router_engine") is None
    assert close_calls == {"sync": 1, "async": 1}


@pytest.mark.unit
def test_hydrate_publication_failure_clears_old_and_staged_runtime(monkeypatch):
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")
    vector_session = importlib.import_module("src.ui.vector_session")

    class _Client:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class _Router:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    old_client = _Client()
    old_router = _Router()
    old_resource = page.VectorIndexResource("old", client=old_client)
    vector_session.replace_session_runtime(
        st.session_state,
        old_resource,
        old_router,
        runtime_generation=page.settings.cache_version,
        state_updates={"_snapshot_loaded_id": "old-snapshot"},
    )

    class _FailingState(dict[str, object]):
        armed = True

        def __setitem__(self, key: str, value: object) -> None:
            super().__setitem__(key, value)
            if self.armed and key == "router_engine":
                self.armed = False
                raise RuntimeError("publication failed")

    state = _FailingState(st.session_state)
    monkeypatch.setattr(page.st, "session_state", state)
    new_client = _Client()
    vector_store = SimpleNamespace(client=new_client)
    vector_index = SimpleNamespace(vector_store=vector_store)
    new_router = _Router()
    monkeypatch.setattr(page, "load_manifest", lambda _path: _snapshot_manifest())
    monkeypatch.setattr(page, "load_vector_index", lambda _path: vector_index)
    monkeypatch.setattr(page, "load_property_graph_index", lambda _path: None)
    monkeypatch.setattr(page, "build_router_engine", lambda *_a, **_k: new_router)
    with pytest.raises(
        RuntimeError, match="Activated snapshot router construction failed"
    ):
        page._hydrate_router_from_snapshot(Path("/tmp/new-snapshot"))

    assert old_client.close_calls == 1
    assert old_router.close_calls == 1
    assert new_client.close_calls == 1
    assert new_router.close_calls == 1
    assert state["vector_index"] is None
    assert state["router_engine"] is None
    assert "_snapshot_loaded_id" not in state


@pytest.mark.unit
def test_load_latest_snapshot_policies(monkeypatch, tmp_path):
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    import streamlit as st  # type: ignore

    from src.config import settings

    st.session_state.clear()

    # ignore policy → no action
    monkeypatch.setattr(settings.graphrag_cfg, "autoload_policy", "ignore")
    page._load_latest_snapshot_into_session()
    assert st.session_state.get("router_engine") is None
    assert st.session_state.get("vector_index") is None
    assert "_snapshot_loaded_id" not in st.session_state

    # latest verified, non-stale policy
    d = tmp_path / "storage" / "SID"
    d.mkdir(parents=True)
    monkeypatch.setattr(settings.graphrag_cfg, "autoload_policy", "latest_non_stale")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(page, "latest_snapshot_dir", lambda: d)
    monkeypatch.setattr(page, "load_manifest", lambda _p: _snapshot_manifest())
    monkeypatch.setattr(page, "compute_staleness", lambda *_a, **_k: False)

    # Ensure hydrate path callable but side effects minimal
    monkeypatch.setattr(
        page,
        "_hydrate_router_from_snapshot_quiesced",
        lambda p: st.session_state.__setitem__("router_engine", "R"),
    )
    page._load_latest_snapshot_into_session()
    page._load_latest_snapshot_into_session()
    assert st.session_state.get("router_engine") == "R"

    class _OldRouter:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    old = _OldRouter()
    st.session_state["router_engine"] = old
    monkeypatch.setattr(page, "compute_staleness", lambda *_a, **_k: True)
    page._load_latest_snapshot_into_session()
    assert old.closed == 1
    assert st.session_state["router_engine"] is None


@pytest.mark.unit
def test_close_memory_store_delegates_to_native_helper(monkeypatch):
    import importlib

    page = importlib.import_module("src.pages.01_chat")

    store = object()
    closed: list[object] = []
    monkeypatch.setattr(page, "close_memory_store", closed.append)
    page._close_memory_store(store)
    assert closed == [store]
    info = page._get_memory_store._info  # type: ignore[attr-defined]
    assert info.on_release is page._close_memory_store


@pytest.mark.unit
def test_memory_store_initializes_embedding_before_opening_index(monkeypatch):
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    events: list[str] = []
    expected_store = object()

    monkeypatch.setattr(
        page,
        "setup_llamaindex",
        lambda: events.append("setup"),
    )

    def _adapter() -> object:
        events.append("adapter")
        return object()

    def _open(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        events.append("open")
        return expected_store

    monkeypatch.setattr(page, "LlamaIndexEmbeddingsAdapter", _adapter)
    monkeypatch.setattr(page, "open_memory_store", _open)

    store = page._get_memory_store.__wrapped__()  # type: ignore[attr-defined]

    assert store is expected_store
    assert events == ["setup", "adapter", "open"]


@pytest.mark.unit
@pytest.mark.parametrize("resource_kind", ["chat_session", "memory_store"])
def test_cached_live_reader_blocks_release_until_lease_exits(
    monkeypatch: pytest.MonkeyPatch,
    resource_kind: str,
) -> None:
    """Cache release cannot close a DB/store held by a foreground reader."""
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    chat_sessions = importlib.import_module("src.ui.chat_sessions")
    background_jobs = importlib.import_module("src.ui.background_jobs")
    manager = background_jobs.JobManager()
    entered = threading.Event()
    release_reader = threading.Event()
    errors: list[BaseException] = []

    class _Resource:
        closed = False

        def close(self) -> None:
            self.closed = True

    resource = _Resource()
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    if resource_kind == "chat_session":
        monkeypatch.setattr(page, "get_chat_db_conn", lambda: resource)
        activity = page._chat_session_activity
        release_callback = chat_sessions._close_chat_db_conn
    else:
        monkeypatch.setattr(page, "_get_memory_store", lambda: resource)
        monkeypatch.setattr(page, "close_memory_store", lambda store: store.close())
        activity = page._memory_store_activity
        release_callback = page._close_memory_store

    def _reader() -> None:
        try:
            with activity() as current:
                assert current is resource
                assert not resource.closed
                entered.set()
                assert release_reader.wait(5)
                assert not resource.closed
        except BaseException as exc:  # pragma: no cover - surfaced in caller
            errors.append(exc)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    try:
        assert entered.wait(2)
        with (
            pytest.raises(background_jobs.ForegroundRuntimeConflictError),
            manager.admission_quiescence(),
        ):
            pytest.fail("maintenance entered while a cached resource was leased")
        assert not resource.closed
        release_reader.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert errors == []

        with manager.admission_quiescence():
            release_callback(resource)
        assert resource.closed
    finally:
        release_reader.set()
        thread.join(timeout=2)
        manager.shutdown()


@pytest.mark.unit
@pytest.mark.parametrize("resource_kind", ["chat_session", "memory_store"])
def test_cached_live_resource_rejects_reader_during_maintenance_and_releases_lease(
    monkeypatch: pytest.MonkeyPatch,
    resource_kind: str,
) -> None:
    """Maintenance wins cleanly and reader exceptions release their lease."""
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    chat_sessions = importlib.import_module("src.ui.chat_sessions")
    background_jobs = importlib.import_module("src.ui.background_jobs")
    manager = background_jobs.JobManager()
    acquisitions = 0

    class _Resource:
        closed = False

        def close(self) -> None:
            self.closed = True

    old_resource = _Resource()
    current_resource = _Resource()

    def _get_current() -> _Resource:
        nonlocal acquisitions
        acquisitions += 1
        return current_resource

    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    if resource_kind == "chat_session":
        monkeypatch.setattr(page, "get_chat_db_conn", _get_current)
        activity = page._chat_session_activity
        release_callback = chat_sessions._close_chat_db_conn
    else:
        monkeypatch.setattr(page, "_get_memory_store", _get_current)
        monkeypatch.setattr(page, "close_memory_store", lambda store: store.close())
        activity = page._memory_store_activity
        release_callback = page._close_memory_store

    def _read_and_fail() -> None:
        with activity() as current:
            assert current is current_resource
            raise RuntimeError("reader failed")

    try:
        with manager.admission_quiescence():
            release_callback(old_resource)
            with pytest.raises(background_jobs.JobAdmissionPausedError), activity():
                pytest.fail("reader acquired a cache resource during maintenance")
        assert old_resource.closed
        assert acquisitions == 0

        with pytest.raises(RuntimeError, match="reader failed"):
            _read_and_fail()
        assert acquisitions == 1
        assert not manager.activity_snapshot().foreground_runtime_active

        with manager.admission_quiescence():
            release_callback(current_resource)
        assert current_resource.closed
    finally:
        manager.shutdown()


@pytest.mark.unit
def test_coordinator_owns_async_checkpointer_lifecycle(monkeypatch):
    """Chat delegates deterministic coordinator lifecycle to the runtime owner."""
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    runtime = importlib.import_module("src.ui.chat_runtime")
    coordinator_module = importlib.import_module("src.agents.coordinator")
    memory_store = object()
    closed: list[object] = []

    class _Coordinator:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs

        def close(self) -> None:
            closed.append(self)

    runtime.invalidate_coordinator()
    monkeypatch.setattr(coordinator_module, "MultiAgentCoordinator", _Coordinator)
    monkeypatch.setattr(page, "_get_memory_store", lambda: memory_store)

    coordinator = page._get_coordinator()

    assert coordinator.kwargs == {
        "checkpointer_path": page.settings.chat.sqlite_path,
        "store": memory_store,
    }
    assert closed == []
    runtime.invalidate_coordinator()
    assert closed == [coordinator]


@pytest.mark.unit
def test_analysis_sidebar_disables_run_during_maintenance(
    monkeypatch: pytest.MonkeyPatch,
    clean_streamlit_session: dict[str, list[str] | int],
) -> None:
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    disabled: dict[str, bool] = {}

    class _Column:
        def button(self, label: str, **kwargs: object) -> bool:
            disabled[label] = bool(kwargs.get("disabled"))
            return False

    manager = SimpleNamespace(
        activity_snapshot=lambda: SimpleNamespace(maintenance_active=True)
    )
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(
        page,
        "_get_analysis_sidebar_inputs",
        lambda: ("combined", [], "question"),
    )
    monkeypatch.setattr(page.st, "columns", lambda _count: [_Column(), _Column()])

    page._render_analysis_sidebar(owner_id="owner")

    assert disabled == {"Run analysis": True, "Cancel": True}
    assert clean_streamlit_session["infos"] == [
        "Runtime maintenance is in progress. Analysis is unavailable."
    ]


@pytest.mark.unit
def test_analysis_sidebar_handles_maintenance_admission_race(
    monkeypatch: pytest.MonkeyPatch,
    clean_streamlit_session: dict[str, list[str] | int],
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")

    class _Column:
        def __init__(self, clicked: bool) -> None:
            self._clicked = clicked

        def button(self, _label: str, **_kwargs: object) -> bool:
            return self._clicked

    class _Manager:
        activity_snapshot = staticmethod(
            lambda: SimpleNamespace(maintenance_active=False)
        )

        @contextlib.contextmanager
        def foreground_runtime_activity(self):  # type: ignore[no-untyped-def]
            raise page.JobAdmissionPausedError("maintenance won")
            yield

        def start_job(self, **_kwargs: object) -> str:
            pytest.fail("job submission ran without a foreground lease")

    monkeypatch.setattr(page, "get_job_manager", _Manager)
    monkeypatch.setattr(
        page,
        "_get_analysis_sidebar_inputs",
        lambda: ("combined", [], "question"),
    )
    monkeypatch.setattr(
        page.st,
        "columns",
        lambda _count: [_Column(True), _Column(False)],
    )
    st.session_state["vector_index"] = object()

    page._render_analysis_sidebar(owner_id="owner")

    assert "analysis_job_id" not in st.session_state
    assert clean_streamlit_session["warnings"] == [
        "Runtime maintenance is in progress. Try analysis again shortly."
    ]


@pytest.mark.unit
def test_analysis_runtime_capture_transitions_atomically_to_background_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")
    background_jobs = importlib.import_module("src.ui.background_jobs")
    manager = background_jobs.JobManager(max_workers=1)
    started = threading.Event()
    release = threading.Event()

    class _Column:
        def __init__(self, clicked: bool) -> None:
            self._clicked = clicked

        def button(self, _label: str, **_kwargs: object) -> bool:
            return self._clicked

    def _work(*_args: object, **_kwargs: object) -> object:
        started.set()
        release.wait(timeout=5)
        return object()

    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(
        page,
        "_get_analysis_sidebar_inputs",
        lambda: ("combined", [], "question"),
    )
    monkeypatch.setattr(
        page.st,
        "columns",
        lambda _count: [_Column(True), _Column(False)],
    )
    monkeypatch.setattr(page, "_run_analysis_job_work", _work)
    st.session_state["vector_index"] = object()

    try:
        page._render_analysis_sidebar(owner_id="owner")
        assert started.wait(timeout=2)
        with (
            pytest.raises(background_jobs.JobConflictError),
            manager.admission_quiescence(),
        ):
            pytest.fail("maintenance overlapped the captured analysis runtime")
    finally:
        release.set()
        job_id = st.session_state.get("analysis_job_id")
        if isinstance(job_id, str):
            assert manager.wait_for_completion(job_id, owner_id="owner") == "succeeded"
        manager.shutdown()


@pytest.mark.unit
def test_analysis_success_transfers_result_then_releases_manager_owners(
    monkeypatch: pytest.MonkeyPatch,
    clean_streamlit_session: dict[str, list[str] | int],
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.analysis.models import AnalysisResult
    from src.ui.background_jobs import JobManager

    page = importlib.import_module("src.pages.01_chat")
    manager = JobManager(max_workers=1)
    result = AnalysisResult(
        mode="combined",
        per_doc=[],
        combined="answer",
        reduce=None,
        warnings=[],
        auto_decision_reason=None,
    )
    job_id = manager.start_job(
        owner_id="owner",
        fn=lambda _cancel, _report: result,
    )
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    _interrupt_on_app_rerun(monkeypatch, clean_streamlit_session)

    try:
        assert manager.wait_for_completion(job_id, owner_id="owner") == "succeeded"
        st.session_state["analysis_job_id"] = job_id

        with pytest.raises(_AppRerunRequestedError):
            page._render_analysis_job_panel.__wrapped__(owner_id="owner")

        assert st.session_state["analysis_last_result"] is result
        assert st.session_state[page._ANALYSIS_COMPLETED_JOB_KEY] == job_id
        assert st.session_state[page._ANALYSIS_TERMINAL_NOTICE_KEY] == {
            "status": "succeeded",
            "message": "Analysis completed.",
        }
        assert "analysis_job_id" not in st.session_state
        assert manager.get(job_id, owner_id="owner") is None
        assert job_id not in manager._jobs
        assert job_id not in manager._futures
        assert clean_streamlit_session["successes"] == []
        assert clean_streamlit_session["reruns"] == 1
        assert clean_streamlit_session["rerun_scopes"] == ["app"]

        page._render_analysis_job_panel.__wrapped__(owner_id="owner")
        page._render_analysis_terminal_notice()
        page._render_analysis_terminal_notice()

        assert clean_streamlit_session["successes"] == ["Analysis completed."]
    finally:
        manager.shutdown()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "capture_key", "message"),
    [
        ("failed", "errors", "Analysis failed."),
        ("canceled", "warnings", "Analysis canceled."),
    ],
)
def test_analysis_terminal_failure_and_cancel_are_visible_once(
    monkeypatch: pytest.MonkeyPatch,
    clean_streamlit_session: dict[str, list[str] | int],
    status: str,
    capture_key: str,
    message: str,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")
    job_id = f"job-{status}"

    class _Manager:
        def get(self, _job_id: str, *, owner_id: str) -> SimpleNamespace:
            assert (_job_id, owner_id) == (job_id, "owner")
            return SimpleNamespace(status=status, result=None, error="redacted")

        def drain_progress(self, _job_id: str, *, owner_id: str) -> list[object]:
            assert (_job_id, owner_id) == (job_id, "owner")
            return []

        def consume_terminal(self, _job_id: str, *, owner_id: str) -> bool:
            assert (_job_id, owner_id) == (job_id, "owner")
            return True

    monkeypatch.setattr(page, "get_job_manager", _Manager)
    _interrupt_on_app_rerun(monkeypatch, clean_streamlit_session)
    st.session_state["analysis_job_id"] = job_id

    with pytest.raises(_AppRerunRequestedError):
        page._render_analysis_job_panel.__wrapped__(owner_id="owner")

    assert clean_streamlit_session[capture_key] == []
    assert clean_streamlit_session["rerun_scopes"] == ["app"]
    assert st.session_state[page._ANALYSIS_TERMINAL_NOTICE_KEY] == {
        "status": status,
        "message": message,
    }

    page._render_analysis_job_panel.__wrapped__(owner_id="owner")
    page._render_analysis_terminal_notice()
    page._render_analysis_terminal_notice()

    assert clean_streamlit_session[capture_key] == [message]
    assert "analysis_job_id" not in st.session_state


@pytest.mark.unit
def test_analysis_consume_retry_does_not_republish_result_or_notice(
    monkeypatch: pytest.MonkeyPatch,
    clean_streamlit_session: dict[str, list[str] | int],
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.analysis.models import AnalysisResult

    page = importlib.import_module("src.pages.01_chat")
    result = AnalysisResult(
        mode="combined",
        per_doc=[],
        combined="answer",
        reduce=None,
        warnings=[],
        auto_decision_reason=None,
    )
    job_id = "job-pending-consume"

    class _Manager:
        def __init__(self) -> None:
            self.consume_results = iter((False, False, True))
            self.drain_calls = 0

        def get(self, _job_id: str, *, owner_id: str) -> SimpleNamespace:
            assert (_job_id, owner_id) == (job_id, "owner")
            return SimpleNamespace(status="succeeded", result=result, error=None)

        def drain_progress(self, _job_id: str, *, owner_id: str) -> list[object]:
            assert (_job_id, owner_id) == (job_id, "owner")
            self.drain_calls += 1
            return []

        def consume_terminal(self, _job_id: str, *, owner_id: str) -> bool:
            assert (_job_id, owner_id) == (job_id, "owner")
            return next(self.consume_results)

    manager = _Manager()
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    _interrupt_on_app_rerun(monkeypatch, clean_streamlit_session)
    st.session_state["analysis_job_id"] = job_id

    with pytest.raises(_AppRerunRequestedError):
        page._render_analysis_job_panel.__wrapped__(owner_id="owner")

    assert st.session_state["analysis_job_id"] == job_id
    assert st.session_state["analysis_last_result"] is result
    assert clean_streamlit_session["successes"] == []
    assert clean_streamlit_session["rerun_scopes"] == ["app"]

    page._render_analysis_job_panel.__wrapped__(owner_id="owner")
    page._render_analysis_terminal_notice()

    assert clean_streamlit_session["successes"] == ["Analysis completed."]

    with pytest.raises(_AppRerunRequestedError):
        page._render_analysis_job_panel.__wrapped__(owner_id="owner")

    assert "analysis_job_id" not in st.session_state
    assert st.session_state["analysis_last_result"] is result
    assert manager.drain_calls == 1
    assert clean_streamlit_session["successes"] == ["Analysis completed."]
    assert clean_streamlit_session["rerun_scopes"] == ["app", "app"]
    assert page._ANALYSIS_TERMINAL_NOTICE_KEY not in st.session_state

    page._render_analysis_job_panel.__wrapped__(owner_id="owner")
    page._render_analysis_terminal_notice()
    assert clean_streamlit_session["successes"] == ["Analysis completed."]


@pytest.mark.unit
def test_analysis_missing_state_clears_tracking_and_reenables_run_control(
    monkeypatch: pytest.MonkeyPatch,
    clean_streamlit_session: dict[str, list[str] | int],
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")
    disabled: dict[str, bool] = {}

    class _Column:
        def button(self, label: str, **kwargs: object) -> bool:
            disabled[label] = bool(kwargs.get("disabled"))
            return False

    class _Manager:
        def get(self, _job_id: str, *, owner_id: str) -> None:
            assert (_job_id, owner_id) == ("missing-job", "owner")
            return None

        @staticmethod
        def activity_snapshot() -> SimpleNamespace:
            return SimpleNamespace(maintenance_active=False)

    manager = _Manager()
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    _interrupt_on_app_rerun(monkeypatch, clean_streamlit_session)
    monkeypatch.setattr(
        page,
        "_get_analysis_sidebar_inputs",
        lambda: ("combined", [], "question"),
    )
    monkeypatch.setattr(page.st, "columns", lambda _count: [_Column(), _Column()])
    st.session_state["analysis_job_id"] = "missing-job"
    st.session_state["analysis_cancel_requested_id"] = "missing-job"
    durable_result = object()
    st.session_state["analysis_last_result"] = durable_result
    st.session_state[page._ANALYSIS_COMPLETED_JOB_KEY] = "missing-job"
    st.session_state[page._ANALYSIS_TERMINAL_NOTICE_KEY] = {
        "status": "failed",
        "message": "Analysis failed.",
    }

    with pytest.raises(_AppRerunRequestedError):
        page._render_analysis_job_panel.__wrapped__(owner_id="owner")

    assert "analysis_job_id" not in st.session_state
    assert "analysis_cancel_requested_id" not in st.session_state
    assert clean_streamlit_session["rerun_scopes"] == ["app"]
    assert st.session_state["analysis_last_result"] is durable_result
    assert st.session_state[page._ANALYSIS_COMPLETED_JOB_KEY] == "missing-job"

    page._render_analysis_job_panel.__wrapped__(owner_id="owner")
    page._render_analysis_terminal_notice()

    assert clean_streamlit_session["errors"] == ["Analysis failed."]

    page._render_analysis_sidebar(owner_id="owner")

    assert disabled == {"Run analysis": False, "Cancel": True}


@pytest.mark.unit
def test_set_last_sources_for_render_sets_session_state():
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")
    st.session_state.clear()
    sources = [{"content": "c", "metadata": {"doc_id": "d"}}]
    page._set_last_sources_for_render(sources, thread_id="t1")
    assert st.session_state["active_thread_id"] == "t1"
    assert st.session_state["last_sources"] == sources


@pytest.mark.unit
def test_render_sources_fragment_renders_text_and_image(
    monkeypatch, clean_streamlit_session
):
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")

    class _Store:
        def resolve_path(self, _ref):  # type: ignore[no-untyped-def]
            return Path("img.png")

    monkeypatch.setattr(page.ArtifactStore, "from_settings", lambda _s: _Store())
    monkeypatch.setattr(st, "slider", lambda *_a, **_k: 2, raising=False)

    st.session_state["active_thread_id"] = "t1"
    st.session_state["last_sources"] = [
        {
            "metadata": {
                "modality": "pdf_page_image",
                "doc_id": "d1",
                "page_no": 1,
                "thumbnail_artifact_id": "a",
                "thumbnail_artifact_suffix": ".png",
            }
        },
        {"content": "hello", "metadata": {"doc_id": "d1"}},
    ]

    page._render_sources_fragment.__wrapped__()  # type: ignore[attr-defined]
    assert clean_streamlit_session["images"]
    assert clean_streamlit_session["markdown"]


@pytest.mark.unit
def test_render_memory_sidebar_puts_and_reruns(monkeypatch, clean_streamlit_session):
    import importlib

    import streamlit as st  # type: ignore

    from src.agents.tools.memory import memory_id
    from src.persistence.checkpoint_identity import memory_namespace

    page = importlib.import_module("src.pages.01_chat")

    seen: dict[str, object] = {}

    class _Store:
        def get(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return None

        def put(self, ns, key, value, index=None):  # type: ignore[no-untyped-def]
            seen["put"] = (ns, key, value, index)

        def search(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return []

        def delete(self, *_a, **_k):  # type: ignore[no-untyped-def]
            raise AssertionError("delete should not be called")

    monkeypatch.setattr(page, "_get_memory_store", lambda: _Store())

    def _text_input(label, key=None, **_k):  # type: ignore[no-untyped-def]
        if key == "memory_add":
            return "remember me"
        if key == "memory_search":
            return ""
        return ""

    def _button(label, key=None, **_k):  # type: ignore[no-untyped-def]
        return key == "memory_save"

    monkeypatch.setattr(st, "text_input", _text_input, raising=False)
    monkeypatch.setattr(st, "button", _button, raising=False)

    page._render_memory_sidebar("u1", "t1")
    assert "put" in seen
    namespace, key, value, index = cast(tuple, seen["put"])
    assert namespace == memory_namespace(user_id="u1", thread_id="t1")
    assert key == memory_id("remember me", "fact")
    assert value["origin"] == "explicit"
    assert index == ["content"]
    assert "_memory_last_saved" not in st.session_state
    assert st.session_state["memory_add"] == ""
    assert int(clean_streamlit_session["reruns"]) == 1


@pytest.mark.unit
def test_ensure_router_engine_sets_none_and_hydrates(
    monkeypatch, clean_streamlit_session
):
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")

    st.session_state.clear()
    monkeypatch.setattr(
        page,
        "_load_latest_snapshot_into_session",
        lambda: (_ for _ in ()).throw(RuntimeError("x")),
    )
    monkeypatch.setattr(page, "latest_snapshot_dir", lambda *_a, **_k: None)
    page._ensure_router_engine()
    assert st.session_state.get("router_engine") is None
    assert any("Autoload skipped" in c for c in clean_streamlit_session["captions"])

    st.session_state.clear()
    monkeypatch.setattr(
        page,
        "_load_latest_snapshot_into_session",
        lambda: replace_session_router(
            st.session_state,
            "R",
            runtime_generation=page.settings.cache_version,
        ),
    )
    page._ensure_router_engine()
    assert st.session_state.get("router_engine") == "R"


@pytest.mark.unit
def test_render_staleness_badge_warns_and_logs(
    monkeypatch, tmp_path: Path, clean_streamlit_session
):
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.01_chat")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    storage = tmp_path / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    snap = storage / "sid"
    snap.mkdir()

    events: list[dict] = []
    monkeypatch.setattr(page, "latest_snapshot_dir", lambda *_a, **_k: snap)
    monkeypatch.setattr(
        page,
        "load_manifest",
        lambda _p: {
            "corpus_hash": "c",
            "config_hash": "d",
            "collections": dict(_PHYSICAL_COLLECTIONS),
        },
    )
    monkeypatch.setattr(page, "_collect_corpus_paths", lambda _p: [])
    monkeypatch.setattr(page, "_current_config_dict", lambda: {})
    monkeypatch.setattr(page, "compute_staleness", lambda *_a, **_k: True)
    monkeypatch.setattr(page, "log_jsonl", lambda ev: events.append(ev))

    page._render_staleness_badge()
    assert page.STALE_TOOLTIP in clean_streamlit_session["warnings"]
    assert events
    assert events[-1].get("snapshot_stale_detected") is True


@pytest.mark.unit
def test_render_staleness_badge_sanitizes_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clean_streamlit_session: dict[str, list[str] | int],
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.01_chat")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    (tmp_path / "storage").mkdir()
    monkeypatch.setattr(
        page,
        "latest_snapshot_dir",
        lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("staleness leaked-secret")
        ),
    )

    page._render_staleness_badge()

    assert clean_streamlit_session["captions"] == ["Staleness check unavailable."]
    assert "leaked-secret" not in str(clean_streamlit_session["captions"])


@pytest.mark.unit
def test_render_chat_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    import streamlit as st  # type: ignore
    from langchain_core.messages import AIMessage, HumanMessage

    page = importlib.import_module("src.pages.01_chat")
    roles: list[str] = []

    def _chat_message(role: str):  # type: ignore[no-untyped-def]
        roles.append(str(role))
        return contextlib.nullcontext()

    monkeypatch.setattr(st, "chat_message", _chat_message, raising=False)
    page._render_chat_history(
        [HumanMessage(content="hi"), AIMessage(content="yo"), object()]
    )

    assert roles == ["user", "assistant"]


@pytest.mark.unit
def test_handle_prompt_preserves_status_persistence_and_sources(
    monkeypatch: pytest.MonkeyPatch,
    clean_streamlit_session: dict[str, list[str] | int],
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")
    monkeypatch.setattr(st, "chat_input", lambda *_a, **_k: "Q", raising=False)

    events, touched, query_kwargs = [], [], []

    def _touch_session(*args: object, **kwargs: object) -> None:
        assert manager.active
        events.append("touch")
        touched.append((args, kwargs))

    monkeypatch.setattr(page, "touch_session", _touch_session)

    def _render_sources() -> None:
        assert not manager.active
        events.append("sources")

    monkeypatch.setattr(page, "_render_sources_fragment", _render_sources)

    class _Manager:
        active = False

        @contextlib.contextmanager
        def foreground_runtime_activity(self):  # type: ignore[no-untyped-def]
            self.active = True
            try:
                yield
            finally:
                self.active = False

    class _Coord:
        def process_query(self, **kwargs: object) -> SimpleNamespace:
            assert manager.active
            events.append("process")
            query_kwargs.append(kwargs)
            return SimpleNamespace(content="A", sources=[{"content": "src"}])

        def list_checkpoints(self, **_k):  # type: ignore[no-untyped-def]
            assert manager.active
            events.append("checkpoints")
            return [{"checkpoint_id": "cp"}]

    manager, coord = _Manager(), _Coord()
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)

    def _current_coord():  # type: ignore[no-untyped-def]
        assert manager.active
        return coord

    monkeypatch.setattr(page, "_get_coordinator", _current_coord)

    @contextlib.contextmanager
    def _spinner(message: str, *, show_time: bool = False):  # type: ignore[no-untyped-def]
        assert message == "Generating response…"
        assert show_time is True
        events.append("spinner-enter")
        try:
            yield
        finally:
            events.append("spinner-exit")

    monkeypatch.setattr(st, "spinner", _spinner, raising=False)
    monkeypatch.setattr(
        page,
        "_get_settings_override",
        lambda: (
            None if manager.active else pytest.fail("override acquired before lease")
        ),
    )
    conn = sqlite3.connect(":memory:")
    try:

        def _current_conn() -> sqlite3.Connection:
            assert manager.active
            return conn

        monkeypatch.setattr(page, "get_chat_db_conn", _current_conn)
        selection = SimpleNamespace(thread_id="t", user_id="u")
        page._handle_chat_prompt(selection)
    finally:
        conn.close()
    assert touched
    assert query_kwargs == [
        {
            "query": "Q",
            "settings_override": None,
            "thread_id": "t",
            "user_id": "u",
        }
    ]
    assert events == [
        "spinner-enter",
        "process",
        "spinner-exit",
        "checkpoints",
        "touch",
        "sources",
    ]
    assert touched == [
        (
            (conn,),
            {"thread_id": "t", "last_checkpoint_id": "cp"},
        )
    ]
    assert "A" in cast(list[str], clean_streamlit_session["markdown"])
    assert st.session_state["last_sources"] == [{"content": "src"}]
    assert not manager.active


@pytest.mark.unit
def test_coordinator_callbacks_reacquire_current_runtime(monkeypatch) -> None:
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    active = False
    acquisitions: list[str] = []

    class _Manager:
        @contextlib.contextmanager
        def foreground_runtime_activity(self):  # type: ignore[no-untyped-def]
            nonlocal active
            active = True
            try:
                yield
            finally:
                active = False

    class _Coord:
        def __init__(self, identity: str) -> None:
            self.identity = identity

        def purge_session(self, **_kwargs: object) -> bool:
            assert active
            acquisitions.append(f"purge:{self.identity}")
            return True

        def fork_from_checkpoint(self, **_kwargs: object) -> str:
            assert active
            acquisitions.append(f"fork:{self.identity}")
            return self.identity

    coordinators = iter((_Coord("first"), _Coord("second")))

    def _current_coord() -> _Coord:
        assert active
        return next(coordinators)

    monkeypatch.setattr(page, "get_job_manager", _Manager)
    monkeypatch.setattr(page, "_get_coordinator", _current_coord)
    conn = sqlite3.connect(":memory:")
    try:

        def _current_conn() -> sqlite3.Connection:
            assert active
            return conn

        monkeypatch.setattr(page, "get_chat_db_conn", _current_conn)
        assert page._purge_chat_session(thread_id="t", user_id="u")
        assert (
            page._fork_chat_checkpoint(thread_id="t", user_id="u", checkpoint_id="c")
            == "second"
        )
    finally:
        conn.close()
    assert acquisitions == ["purge:first", "fork:second"]
    assert not active


@pytest.mark.unit
def test_visual_search_helpers(monkeypatch, clean_streamlit_session):
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")

    # No upload → returns early
    monkeypatch.setattr(st, "file_uploader", lambda *_a, **_k: None, raising=False)
    up, top_k, run = page._visual_search_inputs()
    assert up is None
    assert top_k == 0
    assert run is False

    # Upload path
    upload = object()

    class _Col:
        def number_input(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return 5

        def button(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return True

    monkeypatch.setattr(st, "file_uploader", lambda *_a, **_k: upload, raising=False)
    monkeypatch.setattr(st, "columns", lambda _n: [_Col(), _Col()], raising=False)
    up, top_k, run = page._visual_search_inputs()
    assert up is upload
    assert top_k == 5
    assert run is True

    # Query visual search with stubbed dependencies
    monkeypatch.setattr(
        "src.utils.images.open_untrusted_image", lambda _upload: object()
    )

    mm = importlib.import_module("src.retrieval.multimodal_fusion")

    class _Retriever:
        def __init__(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return None

        def retrieve_by_image(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return [
                SimpleNamespace(
                    node=SimpleNamespace(
                        metadata={
                            "doc_id": "d",
                            "page_no": 1,
                            "thumbnail_artifact_id": "a",
                            "thumbnail_artifact_suffix": ".png",
                        }
                    )
                )
            ]

        def close(self) -> None:
            return None

    monkeypatch.setattr(mm, "ImageSiglipRetriever", _Retriever)
    retriever_calls: list[tuple[str, int]] = []

    def _get_retriever(collection_name: str, cache_version: int) -> _Retriever:
        retriever_calls.append((collection_name, cache_version))
        return _Retriever()

    # Bypass @st.cache_resource while preserving its collection-aware contract.
    monkeypatch.setattr(page, "_get_image_siglip_retriever", _get_retriever)
    st.session_state["_snapshot_collections"] = dict(_PHYSICAL_COLLECTIONS)

    pts = page._query_visual_search(object(), top_k=1)
    assert isinstance(pts, list)
    assert pts
    assert retriever_calls == [("physical-image-v2", page.settings.cache_version)]

    class _Store:
        def resolve_path(self, _ref):  # type: ignore[no-untyped-def]
            return Path("img.png")

    monkeypatch.setattr(page.ArtifactStore, "from_settings", lambda _s: _Store())
    page._render_visual_results(pts, top_k=1)
    assert clean_streamlit_session["images"]

    called: list[tuple] = []
    monkeypatch.setattr(page, "_visual_search_inputs", lambda: (upload, 1, True))
    monkeypatch.setattr(page, "_query_visual_search", lambda *_a, **_k: pts)
    monkeypatch.setattr(
        page, "_render_visual_results", lambda *_a, **_k: called.append((_a, _k))
    )
    page._render_visual_search_sidebar()
    assert called


@pytest.mark.unit
def test_visual_retriever_reader_blocks_release_until_lease_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A visual query owns the foreground lease through retriever use."""
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")
    background_jobs = importlib.import_module("src.ui.background_jobs")
    manager = background_jobs.JobManager()
    entered = threading.Event()
    release_reader = threading.Event()
    errors: list[BaseException] = []

    class _Retriever:
        closed = False

        def retrieve_by_image(self, *_args: object, **_kwargs: object) -> list[object]:
            assert not self.closed
            entered.set()
            assert release_reader.wait(5)
            assert not self.closed
            return []

        def close(self) -> None:
            self.closed = True

    retriever = _Retriever()
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(page, "_get_image_siglip_retriever", lambda *_a: retriever)
    monkeypatch.setattr(
        "src.utils.images.open_untrusted_image", lambda _upload: object()
    )
    st.session_state["_snapshot_collections"] = dict(_PHYSICAL_COLLECTIONS)

    def _reader() -> None:
        try:
            page._query_visual_search(object(), top_k=1)
        except BaseException as exc:  # pragma: no cover - surfaced in caller
            errors.append(exc)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    try:
        assert entered.wait(2)
        with (
            pytest.raises(background_jobs.ForegroundRuntimeConflictError),
            manager.admission_quiescence(),
        ):
            pytest.fail("maintenance entered during visual retrieval")
        assert not retriever.closed
        release_reader.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert errors == []

        with manager.admission_quiescence():
            page._close_image_siglip_retriever(retriever)
        assert retriever.closed
    finally:
        release_reader.set()
        thread.join(timeout=2)
        manager.shutdown()


@pytest.mark.unit
def test_visual_retriever_rejects_reader_during_maintenance_and_releases_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visual retrieval does not acquire stale resources during maintenance."""
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")
    background_jobs = importlib.import_module("src.ui.background_jobs")
    manager = background_jobs.JobManager()
    acquisitions = 0

    class _Retriever:
        def __init__(self) -> None:
            self.closed = False
            self.fail = False

        def retrieve_by_image(self, *_args: object, **_kwargs: object) -> list[object]:
            if self.fail:
                raise RuntimeError("visual query failed")
            return []

        def close(self) -> None:
            self.closed = True

    old_retriever = _Retriever()
    current_retriever = _Retriever()

    def _get_retriever(*_args: object) -> _Retriever:
        nonlocal acquisitions
        acquisitions += 1
        return current_retriever

    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(page, "_get_image_siglip_retriever", _get_retriever)
    monkeypatch.setattr(
        "src.utils.images.open_untrusted_image", lambda _upload: object()
    )
    st.session_state["_snapshot_collections"] = dict(_PHYSICAL_COLLECTIONS)

    try:
        with manager.admission_quiescence():
            page._close_image_siglip_retriever(old_retriever)
            with pytest.raises(background_jobs.JobAdmissionPausedError):
                page._query_visual_search(object(), top_k=1)
        assert old_retriever.closed
        assert acquisitions == 0

        current_retriever.fail = True
        with pytest.raises(RuntimeError, match="visual query failed"):
            page._query_visual_search(object(), top_k=1)
        assert acquisitions == 1
        assert not manager.activity_snapshot().foreground_runtime_active

        with manager.admission_quiescence():
            page._close_image_siglip_retriever(current_retriever)
        assert current_retriever.closed
    finally:
        manager.shutdown()


@pytest.mark.unit
def test_visual_retriever_cache_registers_release_callback() -> None:
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    info = page._get_image_siglip_retriever._info  # type: ignore[attr-defined]

    assert info.on_release is page._close_image_siglip_retriever


@pytest.mark.unit
def test_visual_search_sidebar_reports_maintenance(
    monkeypatch: pytest.MonkeyPatch,
    clean_streamlit_session: dict[str, list[str] | int],
) -> None:
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    monkeypatch.setattr(page, "_visual_search_inputs", lambda: (object(), 1, True))
    monkeypatch.setattr(
        page,
        "_query_visual_search",
        lambda *_a, **_k: (_ for _ in ()).throw(
            page.JobAdmissionPausedError("maintenance")
        ),
    )
    monkeypatch.setattr(
        page,
        "_render_visual_results",
        lambda *_a, **_k: pytest.fail("maintenance result rendered"),
    )

    page._render_visual_search_sidebar()

    assert clean_streamlit_session["warnings"] == [
        "Visual search is unavailable during runtime maintenance."
    ]


@pytest.mark.unit
def test_load_chat_messages_returns_ready_empty_without_rendering(
    monkeypatch: pytest.MonkeyPatch,
    clean_streamlit_session: dict[str, list[str] | int],
) -> None:
    import importlib

    page = importlib.import_module("src.pages.01_chat")

    class _Coord:
        def get_state_values(self, **_k):  # type: ignore[no-untyped-def]
            return {"messages": []}

    @contextlib.contextmanager
    def _activity():  # type: ignore[no-untyped-def]
        yield _Coord()

    monkeypatch.setattr(page, "_coordinator_activity", _activity)
    result = page._load_chat_messages(SimpleNamespace(thread_id="t", user_id="u"))

    assert result == page._ChatHistoryLoadResult(messages=(), status="ready")
    assert clean_streamlit_session["captions"] == []
    assert clean_streamlit_session["infos"] == []
    assert clean_streamlit_session["errors"] == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "state",
    [None, {"messages": None}, {"messages": "not-a-message-list"}],
)
def test_load_chat_messages_malformed_state_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    state: object,
) -> None:
    import importlib

    page = importlib.import_module("src.pages.01_chat")

    class _Coord:
        def get_state_values(self, **_k):  # type: ignore[no-untyped-def]
            return state

    @contextlib.contextmanager
    def _activity():  # type: ignore[no-untyped-def]
        yield _Coord()

    monkeypatch.setattr(page, "_coordinator_activity", _activity)
    result = page._load_chat_messages(
        SimpleNamespace(thread_id="thread", user_id="user")
    )

    assert result.status == "failed"
    assert result.messages == ()
    assert result.fingerprint


@pytest.mark.unit
def test_load_chat_messages_returns_maintenance_without_rendering(
    monkeypatch: pytest.MonkeyPatch,
    clean_streamlit_session: dict[str, list[str] | int],
) -> None:
    import importlib

    page = importlib.import_module("src.pages.01_chat")

    @contextlib.contextmanager
    def _activity():  # type: ignore[no-untyped-def]
        raise page.JobAdmissionPausedError("maintenance")
        yield

    monkeypatch.setattr(page, "_coordinator_activity", _activity)
    result = page._load_chat_messages(SimpleNamespace(thread_id="t", user_id="u"))

    assert result == page._ChatHistoryLoadResult(messages=(), status="maintenance")
    assert clean_streamlit_session["captions"] == []
    assert clean_streamlit_session["infos"] == []
    assert clean_streamlit_session["errors"] == []


@pytest.mark.unit
def test_load_chat_messages_failure_is_sanitized_and_retryable(
    monkeypatch: pytest.MonkeyPatch,
    clean_streamlit_session: dict[str, list[str] | int],
) -> None:
    import importlib

    page = importlib.import_module("src.pages.01_chat")

    @contextlib.contextmanager
    def _activity():  # type: ignore[no-untyped-def]
        raise RuntimeError("database leaked-secret")
        yield

    monkeypatch.setattr(page, "_coordinator_activity", _activity)
    selection = SimpleNamespace(
        thread_id="raw-thread-identifier", user_id="raw-user-identifier"
    )
    result = page._load_chat_messages(selection)

    assert result.messages == ()
    assert result.status == "failed"
    assert result.fingerprint
    assert "leaked-secret" not in result.fingerprint
    assert "raw-thread" not in result.fingerprint
    assert clean_streamlit_session["errors"] == []

    monkeypatch.setattr(page.st, "button", lambda *_a, **_k: True)
    assert page._render_chat_history_result(result) is False
    visible = " ".join(
        str(item)
        for key in ("errors", "captions", "infos")
        for item in clean_streamlit_session[key]  # type: ignore[index]
    )
    assert "Chat history could not be loaded. Please retry." in visible
    assert "leaked-secret" not in visible
    assert "raw-thread-identifier" not in visible
    assert "raw-user-identifier" not in visible
    assert clean_streamlit_session["reruns"] == 1


@pytest.mark.unit
def test_render_chat_history_result_distinguishes_empty_and_maintenance(
    clean_streamlit_session: dict[str, list[str] | int],
) -> None:
    import importlib

    page = importlib.import_module("src.pages.01_chat")

    assert page._render_chat_history_result(
        page._ChatHistoryLoadResult(messages=(), status="ready")
    )
    assert clean_streamlit_session["captions"] == [
        "No messages yet. Ask a question to start this chat."
    ]

    assert not page._render_chat_history_result(
        page._ChatHistoryLoadResult(messages=(), status="maintenance")
    )
    assert clean_streamlit_session["infos"] == [
        "Chat history is unavailable during runtime maintenance."
    ]


@pytest.mark.unit
def test_render_memory_sidebar_delete_flow(monkeypatch, clean_streamlit_session):
    import importlib

    import streamlit as st  # type: ignore

    from src.persistence.checkpoint_identity import memory_namespace

    page = importlib.import_module("src.pages.01_chat")

    class _Item:
        def __init__(self) -> None:
            self.key = "k1"
            self.value = {"content": "hello"}
            self.score = 0.5
            self.namespace = memory_namespace(user_id="u1", thread_id="t1")

    deleted: list[tuple] = []

    class _Store:
        def search(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return [_Item()]

        def delete(self, ns, key):  # type: ignore[no-untyped-def]
            deleted.append((ns, key))

    monkeypatch.setattr(page, "_get_memory_store", lambda: _Store())
    monkeypatch.setattr(st, "text_input", lambda *_a, **_k: "", raising=False)
    monkeypatch.setattr(st, "checkbox", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(
        st,
        "button",
        lambda label, **_k: str(label) == "Delete",
        raising=False,
    )

    page._render_memory_sidebar("u1", "t1")
    assert deleted == [(memory_namespace(user_id="u1", thread_id="t1"), "k1")]
    assert int(clean_streamlit_session["reruns"]) == 1


@pytest.mark.unit
def test_render_memory_sidebar_reports_maintenance_without_acquiring_store(
    monkeypatch: pytest.MonkeyPatch,
    clean_streamlit_session: dict[str, list[str] | int],
) -> None:
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    background_jobs = importlib.import_module("src.ui.background_jobs")
    manager = background_jobs.JobManager()
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(
        page,
        "_get_memory_store",
        lambda: pytest.fail("store acquired during runtime maintenance"),
    )

    try:
        with manager.admission_quiescence():
            page._render_memory_sidebar("u1", "t1")
        assert clean_streamlit_session["infos"] == [
            "Memories are unavailable during runtime maintenance."
        ]
    finally:
        manager.shutdown()


@pytest.mark.unit
def test_hydrate_router_from_snapshot_skips_when_already_loaded(monkeypatch):
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")

    st.session_state.clear()
    st.session_state["_snapshot_loaded_id"] = "sid"
    resource = page.VectorIndexResource(object())
    page.replace_session_runtime(
        st.session_state,
        resource,
        object(),
        runtime_generation=page.settings.cache_version,
        state_updates={"_snapshot_loaded_id": "sid"},
    )

    monkeypatch.setattr(
        page,
        "load_vector_index",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not load")),
    )
    page._hydrate_router_from_snapshot(Path("sid"))


@pytest.mark.unit
def test_hydrate_router_from_snapshot_replaces_old_router(monkeypatch):
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")

    class _Old:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    old = _Old()
    new = object()
    st.session_state.clear()
    st.session_state["_snapshot_loaded_id"] = "sid"
    replace_session_router(
        st.session_state,
        old,
        runtime_generation=page.settings.cache_version - 1,
    )

    vector_store = SimpleNamespace(client=SimpleNamespace(close=lambda: None))
    vector_index = SimpleNamespace(vector_store=vector_store)
    monkeypatch.setattr(page, "load_manifest", lambda _p: _snapshot_manifest())
    monkeypatch.setattr(page, "load_vector_index", lambda *_a, **_k: vector_index)
    monkeypatch.setattr(page, "load_property_graph_index", lambda *_a, **_k: None)
    monkeypatch.setattr(page, "build_router_engine", lambda *_a, **_k: new)

    page._hydrate_router_from_snapshot(Path("sid"))
    assert old.closed == 1
    assert st.session_state["router_engine"] is new
    assert "graphrag_index" not in st.session_state


@pytest.mark.unit
def test_hydrate_router_from_snapshot_missing_vector_clears_old_router(monkeypatch):
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")

    class _Old:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    old = _Old()
    st.session_state.clear()
    st.session_state["router_engine"] = old
    monkeypatch.setattr(page, "load_manifest", lambda _p: _snapshot_manifest())
    monkeypatch.setattr(page, "load_vector_index", lambda *_a, **_k: None)
    monkeypatch.setattr(page, "load_property_graph_index", lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match="no vector index"):
        page._hydrate_router_from_snapshot(Path("sid"))

    assert old.closed == 1
    assert st.session_state["router_engine"] is None


@pytest.mark.unit
def test_hydrate_router_from_snapshot_load_failure_clears_old_router(monkeypatch):
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")

    class _Old:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    old = _Old()
    st.session_state.clear()
    st.session_state["router_engine"] = old
    monkeypatch.setattr(page, "load_manifest", lambda _p: _snapshot_manifest())
    monkeypatch.setattr(
        page,
        "load_vector_index",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("load failed")),
    )

    with pytest.raises(OSError, match="load failed"):
        page._hydrate_router_from_snapshot(Path("sid"))

    assert old.closed == 1
    assert st.session_state["router_engine"] is None


@pytest.mark.unit
def test_load_latest_snapshot_without_snapshot_returns(monkeypatch):
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    monkeypatch.setattr(
        page.settings.graphrag_cfg,
        "autoload_policy",
        "latest_non_stale",
    )
    monkeypatch.setattr(page, "latest_snapshot_dir", lambda: None)
    page._load_latest_snapshot_into_session()


@pytest.mark.unit
def test_snapshot_autoload_preserves_runtime_during_active_job(
    monkeypatch, tmp_path: Path, clean_streamlit_session
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")
    background_jobs = importlib.import_module("src.ui.background_jobs")
    manager = background_jobs.JobManager(max_workers=1)
    started = threading.Event()
    release = threading.Event()

    def _worker(_cancel, _progress):  # type: ignore[no-untyped-def]
        started.set()
        release.wait(timeout=5)

    job_id = manager.start_job(owner_id="owner", fn=_worker)
    assert started.wait(timeout=2)
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)

    class _Client:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class _Router:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    client = _Client()
    router = _Router()
    resource = page.VectorIndexResource("old-index", client=client)
    page.replace_session_runtime(
        st.session_state,
        resource,
        router,
        runtime_generation=page.settings.cache_version,
        state_updates={"_snapshot_loaded_id": "old"},
    )
    snapshot = tmp_path / "new"
    monkeypatch.setattr(page, "latest_snapshot_dir", lambda: snapshot)
    monkeypatch.setattr(page, "load_manifest", lambda _path: _snapshot_manifest())
    monkeypatch.setattr(page, "compute_staleness", lambda *_a, **_k: False)
    monkeypatch.setattr(
        page,
        "_hydrate_router_from_snapshot_quiesced",
        lambda _path: pytest.fail("hydration must be deferred"),
    )
    try:
        assert page._load_latest_snapshot_into_session() is False
        assert st.session_state["router_engine"] is router
        assert st.session_state["vector_index"] == "old-index"
        assert client.close_calls == 0
        assert router.close_calls == 0
        assert clean_streamlit_session["captions"][-1] == page._SNAPSHOT_REFRESH_BUSY
    finally:
        release.set()
        assert manager.wait_for_completion(job_id, owner_id="owner") == "succeeded"
        manager.shutdown()


@pytest.mark.unit
def test_snapshot_clear_preserves_runtime_during_maintenance(
    monkeypatch, clean_streamlit_session
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")
    background_jobs = importlib.import_module("src.ui.background_jobs")
    manager = background_jobs.JobManager()
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    router = object()
    resource = page.VectorIndexResource("index")
    page.replace_session_runtime(
        st.session_state,
        resource,
        router,
        runtime_generation=page.settings.cache_version,
        state_updates={"_snapshot_loaded_id": "old"},
    )
    try:
        with manager.admission_quiescence():
            assert page._clear_snapshot_runtime() is False
        assert st.session_state["router_engine"] is router
        assert st.session_state["vector_index"] == "index"
        assert not resource.closed
        assert (
            clean_streamlit_session["captions"][-1]
            == page._SNAPSHOT_REFRESH_MAINTENANCE
        )
    finally:
        manager.shutdown()


@pytest.mark.unit
def test_snapshot_clear_reports_foreground_runtime_conflict(
    monkeypatch, clean_streamlit_session
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")
    background_jobs = importlib.import_module("src.ui.background_jobs")
    manager = background_jobs.JobManager()
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    router = object()
    resource = page.VectorIndexResource("index")
    page.replace_session_runtime(
        st.session_state,
        resource,
        router,
        runtime_generation=page.settings.cache_version,
        state_updates={"_snapshot_loaded_id": "old"},
    )
    try:
        with manager.foreground_runtime_activity():
            assert page._clear_snapshot_runtime() is False
        assert st.session_state["router_engine"] is router
        assert not resource.closed
        assert (
            clean_streamlit_session["captions"][-1] == page._SNAPSHOT_REFRESH_FOREGROUND
        )
    finally:
        manager.shutdown()
