"""Unit tests for Chat page helper functions (no Streamlit runtime).

Covers small pure helpers to raise coverage for the chat page.
"""

from __future__ import annotations

import importlib
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.ui.router_session import replace_session_router


def test_get_settings_override_builds_from_session(monkeypatch):  # type: ignore[no-untyped-def]
    mod = importlib.import_module("src.pages.01_chat")

    class _FakeSession(dict):
        pass

    fake_state = _FakeSession()
    replace_session_router(
        fake_state,
        object(),
        runtime_generation=mod.settings.cache_version,
    )
    fake_state["vector_index"] = 1
    fake_state["graphrag_index"] = 3

    st = importlib.import_module("streamlit")
    monkeypatch.setattr(st, "session_state", fake_state, raising=False)

    overrides = mod._get_settings_override()
    assert overrides is not None
    assert overrides == {"router_engine": fake_state["router_engine"]}


def test_memory_namespace_purge_reports_incomplete_delete() -> None:
    mod = importlib.import_module("src.pages.01_chat")
    namespace = ("memories", "user", "thread")
    item = SimpleNamespace(namespace=namespace, key="memory-1")

    class _FailingStore:
        def search(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return [item]

        def delete(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("database is locked")

    result = mod._purge_memory_namespace(_FailingStore(), namespace)

    assert result.deleted == 0
    assert result.failures == 1
    assert result.complete is False


def test_memory_add_relies_on_canonical_idempotent_store_not_global_guard(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    mod = importlib.import_module("src.pages.01_chat")
    st = importlib.import_module("streamlit")
    state = {"_memory_last_saved": "same content"}
    saved = Mock(return_value="memory-id")
    reruns: list[bool] = []
    monkeypatch.setattr(st, "session_state", state, raising=False)
    monkeypatch.setattr(
        st, "text_input", lambda *_a, **_k: "same content", raising=False
    )
    monkeypatch.setattr(st, "button", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(st, "rerun", lambda: reruns.append(True), raising=False)
    monkeypatch.setattr(mod, "save_memory", saved)
    store = object()
    namespace = ("memories", "other-user", "other-thread")

    mod._render_memory_add(store, namespace)

    saved.assert_called_once()
    assert saved.call_args.args[:2] == (store, namespace)
    assert "_memory_last_saved" not in state
    assert state["memory_add"] == ""
    assert reruns == [True]


def test_memory_add_renders_capacity_error_without_clearing_or_rerunning(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    mod = importlib.import_module("src.pages.01_chat")
    st = importlib.import_module("streamlit")
    state = {"memory_add": "keep this"}
    errors: list[str] = []
    reruns: list[bool] = []
    monkeypatch.setattr(st, "session_state", state, raising=False)
    monkeypatch.setattr(st, "text_input", lambda *_a, **_k: "keep this", raising=False)
    monkeypatch.setattr(st, "button", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(st, "error", errors.append, raising=False)
    monkeypatch.setattr(st, "rerun", lambda: reruns.append(True), raising=False)
    monkeypatch.setattr(
        mod,
        "save_memory",
        Mock(side_effect=mod.MemoryCapacityError("memory namespace is at capacity")),
    )

    mod._render_memory_add(object(), ("memories", "user", "session"))

    assert state["memory_add"] == "keep this"
    assert errors == ["Memory limit reached. Delete a memory before adding another."]
    assert reruns == []


def test_memory_add_tombstone_retains_input_and_reports_rejection(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    mod = importlib.import_module("src.pages.01_chat")
    st = importlib.import_module("streamlit")
    memory_tools = importlib.import_module("src.agents.tools.memory")
    namespace = ("memories", "session", "sidebar-tombstone", "thread")
    memory_tools.tombstone_memory_namespace(namespace)
    state = {"memory_add": "keep this"}
    errors: list[str] = []
    reruns: list[bool] = []

    class _UnusedStore:
        def __getattr__(self, name):  # type: ignore[no-untyped-def]
            raise AssertionError(f"store operation should not run: {name}")

    monkeypatch.setattr(st, "session_state", state, raising=False)
    monkeypatch.setattr(st, "text_input", lambda *_a, **_k: "keep this", raising=False)
    monkeypatch.setattr(st, "button", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(st, "error", errors.append, raising=False)
    monkeypatch.setattr(st, "rerun", lambda: reruns.append(True), raising=False)

    mod._render_memory_add(_UnusedStore(), namespace)

    assert state["memory_add"] == "keep this"
    assert errors == ["This memory scope was purged and is no longer writable."]
    assert reruns == []


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("embedding leaked-content"),
        sqlite3.OperationalError("database leaked-content"),
    ],
    ids=["embedding", "sqlite"],
)
def test_memory_add_store_failure_is_sanitized_and_retains_input(
    monkeypatch,
    failure: Exception,
) -> None:  # type: ignore[no-untyped-def]
    mod = importlib.import_module("src.pages.01_chat")
    st = importlib.import_module("streamlit")
    state = {"memory_add": "keep this"}
    errors: list[str] = []
    reruns: list[bool] = []

    class _FailingStore:
        def get(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return None

        def search(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return []

        def put(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise failure

    monkeypatch.setattr(st, "session_state", state, raising=False)
    monkeypatch.setattr(st, "text_input", lambda *_a, **_k: "keep this", raising=False)
    monkeypatch.setattr(st, "button", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(st, "error", errors.append, raising=False)
    monkeypatch.setattr(st, "rerun", lambda: reruns.append(True), raising=False)

    mod._render_memory_add(_FailingStore(), ("memories", "user", "u"))

    assert state["memory_add"] == "keep this"
    assert errors == ["Memory could not be saved. Please retry."]
    assert "leaked-content" not in errors[0]
    assert reruns == []


def test_memory_results_search_failure_is_sanitized(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    mod = importlib.import_module("src.pages.01_chat")
    st = importlib.import_module("streamlit")
    errors: list[str] = []
    captions: list[str] = []

    class _FailingStore:
        def search(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("search leaked-content")

    monkeypatch.setattr(st, "text_input", lambda *_a, **_k: "query", raising=False)
    monkeypatch.setattr(st, "error", errors.append, raising=False)
    monkeypatch.setattr(st, "caption", captions.append, raising=False)

    mod._render_memory_results(_FailingStore(), ("memories", "user", "u"))

    assert errors == ["Memories could not be loaded. Please retry."]
    assert "leaked-content" not in errors[0]
    assert captions == []


@pytest.mark.parametrize(
    "delete_outcome", [False, RuntimeError("delete leaked-content")]
)
def test_memory_results_delete_failure_keeps_confirmation_and_does_not_rerun(
    monkeypatch,
    delete_outcome: object,
) -> None:  # type: ignore[no-untyped-def]
    mod = importlib.import_module("src.pages.01_chat")
    st = importlib.import_module("streamlit")
    namespace = ("memories", "user", "u")
    item = SimpleNamespace(
        namespace=namespace,
        key="memory-1",
        value={"content": "hello"},
        score=None,
    )
    confirm_key = "mem_del_confirm__memory-1"
    state = {confirm_key: True}
    errors: list[str] = []
    reruns: list[bool] = []

    class _Store:
        def search(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return [item]

        def delete(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            if isinstance(delete_outcome, Exception):
                raise delete_outcome

    if delete_outcome is False:
        monkeypatch.setattr(mod, "delete_memory", Mock(return_value=False))
    monkeypatch.setattr(st, "session_state", state, raising=False)
    monkeypatch.setattr(st, "text_input", lambda *_a, **_k: "", raising=False)
    monkeypatch.setattr(st, "checkbox", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(st, "button", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(st, "caption", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "write", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "error", errors.append, raising=False)
    monkeypatch.setattr(st, "rerun", lambda: reruns.append(True), raising=False)

    mod._render_memory_results(_Store(), namespace)

    assert state[confirm_key] is True
    assert errors == ["Memory could not be deleted. Please retry."]
    assert "leaked-content" not in errors[0]
    assert reruns == []


def test_memory_purge_initial_search_failure_is_incomplete() -> None:
    mod = importlib.import_module("src.pages.01_chat")

    class _FailingStore:
        def search(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("search failed")

    result = mod._purge_memory_namespace(
        _FailingStore(),
        ("memories", "user", "initial-search-failure"),
    )

    assert result == mod._MemoryPurgeResult(deleted=0, failures=1, complete=False)


def test_memory_purge_later_batch_search_failure_is_incomplete() -> None:
    mod = importlib.import_module("src.pages.01_chat")
    namespace = ("memories", "user", "batch-search-failure")
    item = SimpleNamespace(namespace=namespace, key="memory-1")

    class _FullBatch(list):
        def __len__(self) -> int:
            return 5000

    class _FailingStore:
        def __init__(self) -> None:
            self.search_calls = 0

        def search(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.search_calls += 1
            if self.search_calls == 1:
                return _FullBatch([item])
            raise RuntimeError("later search failed")

        def delete(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return None

    store = _FailingStore()
    result = mod._purge_memory_namespace(store, namespace)

    assert result == mod._MemoryPurgeResult(deleted=1, failures=1, complete=False)
    assert store.search_calls == 2


def test_incomplete_memory_purge_keeps_confirmation_and_does_not_rerun(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    mod = importlib.import_module("src.pages.01_chat")
    st = importlib.import_module("streamlit")
    session_state = {"mem_purge_confirm__session__u__t": True}
    errors: list[str] = []
    reruns: list[bool] = []
    monkeypatch.setattr(st, "session_state", session_state, raising=False)
    monkeypatch.setattr(st, "checkbox", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(st, "button", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(st, "error", errors.append, raising=False)
    monkeypatch.setattr(st, "rerun", lambda: reruns.append(True), raising=False)
    monkeypatch.setattr(
        mod,
        "_purge_memory_namespace",
        lambda *_a, **_k: mod._MemoryPurgeResult(
            deleted=0,
            failures=1,
            complete=False,
        ),
    )

    mod._render_memory_purge(
        store=object(),
        ns=("memories", "user", "thread"),
        scope="session",
        user_id="u",
        thread_id="t",
    )

    assert session_state["mem_purge_confirm__session__u__t"] is True
    assert errors
    assert "incomplete" in errors[0].lower()
    assert reruns == []
