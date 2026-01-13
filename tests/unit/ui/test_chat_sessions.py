from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from src.ui import chat_sessions as cs

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _stub_streamlit(monkeypatch):
    import streamlit as st  # type: ignore

    st.session_state.clear()
    monkeypatch.setattr(st, "query_params", {}, raising=False)
    monkeypatch.setattr(st, "sidebar", contextlib.nullcontext(), raising=False)
    monkeypatch.setattr(
        st,
        "spinner",
        lambda _msg: contextlib.nullcontext(),
        raising=False,
    )
    monkeypatch.setattr(st, "rerun", lambda: None, raising=False)

    # UI calls are no-ops by default; tests can override per-case.
    monkeypatch.setattr(st, "subheader", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "caption", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "divider", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "selectbox", lambda *_a, **_k: "", raising=False)
    monkeypatch.setattr(st, "button", lambda *_a, **_k: False, raising=False)
    monkeypatch.setattr(st, "checkbox", lambda *_a, **_k: False, raising=False)
    monkeypatch.setattr(st, "text_input", lambda *_a, **_k: "", raising=False)
    monkeypatch.setattr(
        st,
        "columns",
        lambda n: [SimpleNamespace(button=lambda *_a, **_k: False) for _ in range(n)],
        raising=False,
    )
    return st


def test_get_or_init_user_id_defaults_to_local() -> None:
    import streamlit as st  # type: ignore

    assert cs._get_or_init_user_id() == "local"  # type: ignore[attr-defined]
    assert st.session_state["chat_user_id"] == "local"

    st.session_state["chat_user_id"] = "u1"
    assert cs._get_or_init_user_id() == "u1"  # type: ignore[attr-defined]


def test_seed_from_query_params_sets_session_state() -> None:
    import streamlit as st  # type: ignore

    st.query_params.update({"chat": "t1", "branch": "c1"})
    cs._maybe_seed_from_query_params()  # type: ignore[attr-defined]
    assert st.session_state["chat_thread_id"] == "t1"
    assert st.session_state["chat_resume_checkpoint_id"] == "c1"


def test_seed_from_query_params_accepts_checkpoint_backcompat() -> None:
    import streamlit as st  # type: ignore

    st.query_params.update({"chat": "t1", "checkpoint": "c1"})
    cs._maybe_seed_from_query_params()  # type: ignore[attr-defined]
    assert st.session_state["chat_thread_id"] == "t1"
    assert st.session_state["chat_resume_checkpoint_id"] == "c1"


def test_seed_from_query_params_no_query_params_attr(monkeypatch) -> None:
    import streamlit as st  # type: ignore

    monkeypatch.delattr(st, "query_params", raising=False)
    monkeypatch.setattr(
        st,
        "__getattr__",
        lambda name: (
            (_ for _ in ()).throw(RuntimeError("no query params"))
            if name == "query_params"
            else (_ for _ in ()).throw(AttributeError(name))
        ),
        raising=False,
    )
    cs._maybe_seed_from_query_params()  # type: ignore[attr-defined]
    assert "chat_thread_id" not in st.session_state


def test_render_session_sidebar_normalizes_resume_id(monkeypatch) -> None:
    import streamlit as st  # type: ignore

    active = SimpleNamespace(thread_id="t1", title="a")
    monkeypatch.setattr(cs, "_get_or_init_user_id", lambda: "local")
    monkeypatch.setattr(cs, "ensure_active_session", lambda _c: active)
    monkeypatch.setattr(cs, "list_sessions", lambda _c: [active])
    monkeypatch.setattr(cs, "_render_session_selector", lambda *_a, **_k: "t1")
    monkeypatch.setattr(cs, "_render_new_delete_controls", lambda *_a, **_k: None)
    monkeypatch.setattr(cs, "_handle_rename", lambda *_a, **_k: None)
    monkeypatch.setattr(cs, "_handle_purge", lambda *_a, **_k: None)

    st.session_state.pop("chat_resume_checkpoint_id", None)
    sel = cs.render_session_sidebar(conn=object())  # type: ignore[arg-type]
    assert sel.resume_checkpoint_id is None

    st.session_state["chat_resume_checkpoint_id"] = "c1"
    sel = cs.render_session_sidebar(conn=object())  # type: ignore[arg-type]
    assert sel.resume_checkpoint_id == "c1"


def test_ensure_active_session_creates_when_none(monkeypatch) -> None:
    import streamlit as st  # type: ignore

    created = SimpleNamespace(thread_id="t1", title="New chat")
    monkeypatch.setattr(cs, "list_sessions", lambda _c: [])
    monkeypatch.setattr(cs, "create_session", lambda *, title, conn: created)

    active = cs.ensure_active_session(conn=object())  # type: ignore[arg-type]
    assert active.thread_id == "t1"
    assert st.session_state["chat_thread_id"] == "t1"


def test_render_session_selector_switches_and_touches(monkeypatch) -> None:
    import streamlit as st  # type: ignore

    active = SimpleNamespace(thread_id="t1", title="a")
    sessions = [active, SimpleNamespace(thread_id="t2", title="b")]

    touched: list[str] = []
    reruns = {"n": 0}
    monkeypatch.setattr(
        cs, "touch_session", lambda _c, thread_id: touched.append(thread_id)
    )
    monkeypatch.setattr(st, "selectbox", lambda *_a, **_k: "t2", raising=False)
    monkeypatch.setattr(
        st, "rerun", lambda: reruns.__setitem__("n", reruns["n"] + 1), raising=False
    )

    st.query_params.clear()
    sel = cs._render_session_selector(active, sessions, conn=object())  # type: ignore[arg-type]
    assert sel == "t2"
    assert st.session_state["chat_thread_id"] == "t2"
    assert touched == ["t2"]
    assert st.query_params.get("chat") == "t2"
    assert reruns["n"] == 1


def test_render_time_travel_sidebar_sets_checkpoint(monkeypatch) -> None:
    import streamlit as st  # type: ignore

    monkeypatch.setattr(st, "selectbox", lambda *_a, **_k: "c1", raising=False)
    monkeypatch.setattr(st, "button", lambda *_a, **_k: True, raising=False)
    reruns = {"n": 0}
    monkeypatch.setattr(
        st, "rerun", lambda: reruns.__setitem__("n", reruns["n"] + 1), raising=False
    )

    cs.render_time_travel_sidebar(checkpoints=[{"checkpoint_id": "c1"}])
    assert st.session_state["chat_resume_checkpoint_id"] == "c1"
    assert st.query_params.get("branch") == "c1"
    assert reruns["n"] == 1


def test_render_time_travel_sidebar_no_resume_when_button_false(monkeypatch) -> None:
    import streamlit as st  # type: ignore

    monkeypatch.setattr(st, "selectbox", lambda *_a, **_k: "c1", raising=False)
    monkeypatch.setattr(st, "button", lambda *_a, **_k: False, raising=False)
    cs.render_time_travel_sidebar(checkpoints=[{"checkpoint_id": "c1"}])
    assert "chat_resume_checkpoint_id" not in st.session_state


def test_new_delete_controls_and_rename_and_purge(monkeypatch) -> None:
    import streamlit as st  # type: ignore

    active = SimpleNamespace(thread_id="t1", title="Old")

    # New session path
    created = SimpleNamespace(thread_id="t2", title="New chat")
    monkeypatch.setattr(cs, "create_session", lambda *, title, conn: created)

    def _cols_new(_n: int):  # type: ignore[no-untyped-def]
        return [
            SimpleNamespace(button=lambda *_a, **_k: True),
            SimpleNamespace(button=lambda *_a, **_k: False),
        ]

    reruns = {"n": 0}
    monkeypatch.setattr(st, "columns", _cols_new, raising=False)
    monkeypatch.setattr(st, "checkbox", lambda *_a, **_k: False, raising=False)
    monkeypatch.setattr(
        st, "rerun", lambda: reruns.__setitem__("n", reruns["n"] + 1), raising=False
    )
    cs._render_new_delete_controls(conn=object(), active=active)  # type: ignore[arg-type]
    assert st.session_state["chat_thread_id"] == "t2"
    assert reruns["n"] == 1

    # Delete session path
    deleted: list[str] = []
    monkeypatch.setattr(
        cs, "soft_delete_session", lambda _c, thread_id: deleted.append(thread_id)
    )
    monkeypatch.setattr(
        cs, "list_sessions", lambda _c: [SimpleNamespace(thread_id="t3", title="x")]
    )

    def _cols_delete(_n: int):  # type: ignore[no-untyped-def]
        return [
            SimpleNamespace(button=lambda *_a, **_k: False),
            SimpleNamespace(button=lambda *_a, **_k: True),
        ]

    monkeypatch.setattr(st, "columns", _cols_delete, raising=False)
    monkeypatch.setattr(st, "checkbox", lambda *_a, **_k: True, raising=False)
    cs._render_new_delete_controls(conn=object(), active=active)  # type: ignore[arg-type]
    assert deleted == ["t1"]
    assert st.session_state["chat_thread_id"] == "t3"

    # Rename path
    renamed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cs,
        "rename_session",
        lambda _c, thread_id, title: renamed.append((thread_id, title)),
    )
    monkeypatch.setattr(st, "text_input", lambda *_a, **_k: "New", raising=False)
    monkeypatch.setattr(
        st, "button", lambda label, **_k: str(label) == "Save name", raising=False
    )
    cs._handle_rename(conn=object(), active=active)  # type: ignore[arg-type]
    assert renamed == [("t1", "New")]

    # Purge path
    purged: list[str] = []
    monkeypatch.setattr(
        cs, "purge_session", lambda _c, thread_id: purged.append(thread_id)
    )
    st.session_state["chat_thread_id"] = active.thread_id
    st.session_state["chat_resume_checkpoint_id"] = "c1"
    monkeypatch.setattr(st, "checkbox", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(
        st,
        "button",
        lambda label, **_k: str(label).startswith("Purge session"),
        raising=False,
    )
    cs._handle_purge(conn=object(), active=active)  # type: ignore[arg-type]
    assert purged == ["t1"]
    assert "chat_thread_id" not in st.session_state
    assert "chat_resume_checkpoint_id" not in st.session_state
