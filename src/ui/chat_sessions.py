"""Streamlit helpers for chat session persistence (ADR-057 / SPEC-041)."""

from __future__ import annotations

import contextlib
import sqlite3
from dataclasses import dataclass

import streamlit as st

from src.config.settings import settings
from src.persistence.chat_db import (
    ChatSession,
    create_session,
    ensure_session_registry,
    list_sessions,
    open_chat_db,
    purge_session,
    rename_session,
    soft_delete_session,
    touch_session,
)


@dataclass(frozen=True, slots=True)
class ChatSelection:
    """Resolved chat session selection for this Streamlit run."""

    thread_id: str
    user_id: str
    resume_checkpoint_id: str | None = None


@st.cache_resource(show_spinner=False)
def get_chat_db_conn() -> sqlite3.Connection:
    """Return a cached Chat DB connection and ensure registry tables exist."""
    conn = open_chat_db(settings.chat.sqlite_path, cfg=settings)
    ensure_session_registry(conn)
    return conn


def _get_or_init_user_id() -> str:
    # Local-first UX: user_id is a local namespace key (no auth layer yet).
    if "chat_user_id" not in st.session_state:
        st.session_state["chat_user_id"] = "local"
    return str(st.session_state["chat_user_id"] or "local")


def _maybe_seed_from_query_params() -> None:
    # Query params are cleared when navigating between pages; treat as entry hint.
    try:
        qp = st.query_params
    except Exception:
        return
    if "chat" in qp and "chat_thread_id" not in st.session_state:
        st.session_state["chat_thread_id"] = str(qp.get("chat") or "")
    if "checkpoint" in qp and "chat_resume_checkpoint_id" not in st.session_state:
        st.session_state["chat_resume_checkpoint_id"] = str(qp.get("checkpoint") or "")


def ensure_active_session(conn: sqlite3.Connection) -> ChatSession:
    """Ensure there is an active session selected, creating one if needed."""
    _maybe_seed_from_query_params()
    sessions = list_sessions(conn)
    if not sessions:
        created = create_session(title="New chat", conn=conn)
        st.session_state["chat_thread_id"] = created.thread_id
        return created

    thread_id = str(st.session_state.get("chat_thread_id") or sessions[0].thread_id)
    chosen = next((s for s in sessions if s.thread_id == thread_id), sessions[0])
    st.session_state["chat_thread_id"] = chosen.thread_id
    return chosen


def render_session_sidebar(conn: sqlite3.Connection) -> ChatSelection:
    """Render session management controls and return the current selection."""
    user_id = _get_or_init_user_id()
    active = ensure_active_session(conn)
    sessions = list_sessions(conn)

    with st.sidebar:
        st.subheader("Sessions")
        labels = {s.thread_id: s.title for s in sessions}
        options = [str(s.thread_id) for s in sessions]

        def _format_thread_id(tid: object) -> str:
            tid_str = str(tid)
            return str(labels.get(tid_str) or tid_str)

        sel = st.selectbox(
            "Session",
            options,
            format_func=_format_thread_id,
            index=options.index(active.thread_id) if active.thread_id in options else 0,
        )
        if sel != active.thread_id:
            st.session_state["chat_thread_id"] = sel
            touch_session(conn, thread_id=sel)
            with contextlib.suppress(Exception):
                st.query_params["chat"] = sel
            with st.spinner("Switching session…"):
                st.rerun()

        cols = st.columns(2)
        if cols[0].button("New", use_container_width=True):
            created = create_session(title="New chat", conn=conn)
            st.session_state["chat_thread_id"] = created.thread_id
            with contextlib.suppress(Exception):
                st.query_params["chat"] = created.thread_id
            st.rerun()

        if cols[1].button("Delete", use_container_width=True):
            soft_delete_session(conn, thread_id=active.thread_id)
            # pick another
            remaining = list_sessions(conn)
            if remaining:
                st.session_state["chat_thread_id"] = remaining[0].thread_id
            st.rerun()

        new_title = st.text_input("Rename", value=active.title)
        if new_title and new_title != active.title and st.button("Save name"):
            rename_session(conn, thread_id=active.thread_id, title=new_title)
            st.rerun()

        st.divider()
        st.caption("Danger zone")
        if st.button("Purge session (hard delete)", type="primary"):
            purge_session(conn, thread_id=active.thread_id)
            st.session_state.pop("chat_thread_id", None)
            st.session_state.pop("chat_resume_checkpoint_id", None)
            st.rerun()

    resume = st.session_state.get("chat_resume_checkpoint_id")
    resume_id = str(resume) if resume else None
    if resume_id == "":
        resume_id = None
    return ChatSelection(
        thread_id=str(st.session_state.get("chat_thread_id") or active.thread_id),
        user_id=user_id,
        resume_checkpoint_id=resume_id,
    )


def render_time_travel_sidebar(*, checkpoints: list[dict[str, object]]) -> None:
    """Render time travel controls (sets `chat_resume_checkpoint_id`)."""
    with st.sidebar:
        st.subheader("Time travel")
        st.caption("Resume from a prior checkpoint (creates a fork).")
        ids = [str(c.get("checkpoint_id") or "") for c in checkpoints]
        ids = [x for x in ids if x]
        if not ids:
            st.caption("No checkpoints yet.")
            return
        picked = st.selectbox("Checkpoint", options=ids, index=0)
        if st.button("Resume from checkpoint"):
            st.session_state["chat_resume_checkpoint_id"] = picked
            with contextlib.suppress(Exception):
                st.query_params["checkpoint"] = picked
            st.rerun()
