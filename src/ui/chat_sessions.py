"""Streamlit helpers for chat session persistence (ADR-058 / SPEC-041)."""

from __future__ import annotations

import atexit
import contextlib
import sqlite3
from dataclasses import dataclass
from typing import Protocol

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

# Streamlit runs the UI in a single-threaded model and open_chat_db sets
# check_same_thread=False, so a single cached connection is acceptable. Revisit
# if the app ever runs with concurrent threads/processes.
_chat_db_conn: sqlite3.Connection | None = None
_chat_db_conn_cleanup_registered: bool = False


@dataclass(frozen=True, slots=True)
class ChatSelection:
    """Resolved chat session identifiers for the current application run.

    Attributes:
        thread_id: Unique identifier for the conversation thread.
        user_id: Identifier for the user/tenant namespace.
    """

    thread_id: str
    user_id: str


class ForkableCoordinator(Protocol):
    """Protocol for coordinators supporting checkpoint forking."""

    def fork_from_checkpoint(
        self, *, thread_id: str, user_id: str, checkpoint_id: str
    ) -> str | None:
        """Fork a thread from a checkpoint and return the new checkpoint id."""
        ...


@st.cache_resource(show_spinner=False)
def get_chat_db_conn() -> sqlite3.Connection:
    """Retrieves or initializes a cached SQLite connection for chat metadata.

    Ensures session registry tables exist and registers a cleanup hook for
    connection closure.

    Returns:
        An active SQLite connection instance.
    """
    global _chat_db_conn, _chat_db_conn_cleanup_registered
    if _chat_db_conn is not None:
        _close_chat_db_conn(_chat_db_conn)
    conn = open_chat_db(settings.chat.sqlite_path, cfg=settings)
    ensure_session_registry(conn)
    _chat_db_conn = conn
    if not _chat_db_conn_cleanup_registered:
        atexit.register(_close_current_chat_db_conn)
        _chat_db_conn_cleanup_registered = True
    return conn


def _close_chat_db_conn(conn: sqlite3.Connection) -> None:
    """Safely closes the database connection on application exit.

    Args:
        conn: The SQLite connection to close.
    """
    with contextlib.suppress(Exception):
        conn.close()


def _close_current_chat_db_conn() -> None:
    """Close the currently cached chat DB connection on process exit."""
    global _chat_db_conn
    if _chat_db_conn is None:
        return
    _close_chat_db_conn(_chat_db_conn)
    _chat_db_conn = None


def _get_or_init_user_id() -> str:
    """Resolves the current user identity from session state.

    Defaults to 'local' for environments without an active identity provider.

    Returns:
        The resolved user_id string.
    """
    # Local-first UX: user_id is a local namespace key (no auth layer yet).
    if "chat_user_id" not in st.session_state:
        st.session_state["chat_user_id"] = "local"
    return str(st.session_state["chat_user_id"] or "local")


def _maybe_seed_from_query_params() -> None:
    """Seeds session state from URL query parameters (e.g., ?chat=... &branch=...).

    Supports deep-linking into specific threads and time-travel branches.
    """
    # Query params are cleared when navigating between pages; treat as entry hint.
    try:
        qp = st.query_params
    except Exception:
        return
    if "chat" in qp and "chat_thread_id" not in st.session_state:
        st.session_state["chat_thread_id"] = str(qp.get("chat") or "")
    # SPEC-041 prefers `?branch=<checkpoint_id>` for time-travel links.
    if "chat_time_travel_hint_checkpoint_id" not in st.session_state:
        if "branch" in qp:
            st.session_state["chat_time_travel_hint_checkpoint_id"] = str(
                qp.get("branch") or ""
            )
        elif "checkpoint" in qp:
            # Back-compat: accept older links.
            st.session_state["chat_time_travel_hint_checkpoint_id"] = str(
                qp.get("checkpoint") or ""
            )


def ensure_active_session(conn: sqlite3.Connection) -> ChatSession:
    """Ensures a valid chat thread is selected and active in session state.

    If no sessions exist, a default 'New chat' thread is created.

    Args:
        conn: SQLite connection for metadata persistence.

    Returns:
         The currently active ChatSession record.
    """
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
    """Renders the session management interface in the Streamlit sidebar.

    Provides controls for session selection, creation, and deletion.

    Args:
        conn: SQLite connection for metadata persistence.

    Returns:
        The current chat selection identifiers.
    """
    user_id = _get_or_init_user_id()
    active = ensure_active_session(conn)
    sessions = list_sessions(conn)

    with st.sidebar:
        st.subheader("Sessions")
        _render_session_selector(active, sessions, conn)
        _render_new_delete_controls(conn, active)
        _handle_rename(conn, active)
        st.divider()
        st.caption("Danger zone")
        _handle_purge(conn, active)

    return ChatSelection(
        thread_id=str(st.session_state.get("chat_thread_id") or active.thread_id),
        user_id=user_id,
    )


def _render_session_selector(
    active: ChatSession,
    sessions: list[ChatSession],
    conn: sqlite3.Connection,
) -> str:
    """Renders a selectbox for switching between existing chat sessions.

    Triggers a page rerun and updates query parameters on change.

    Args:
        active: The currently active session.
        sessions: List of available sessions.
        conn: SQLite connection.

    Returns:
        The thread_id of the selected session.
    """
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
        st.session_state.pop("chat_resume_checkpoint_id", None)
        st.session_state.pop("chat_time_travel_hint_checkpoint_id", None)
        with contextlib.suppress(Exception):
            st.query_params.pop("branch", None)
            st.query_params.pop("checkpoint", None)
            st.query_params["chat"] = sel
        with st.spinner("Switching session…"):
            st.rerun()
    return sel


def _render_new_delete_controls(conn: sqlite3.Connection, active: ChatSession) -> None:
    """Renders buttons for creating new sessions and soft-deleting the active one.

    Args:
        conn: SQLite connection.
        active: The currently active session record.
    """
    cols = st.columns(2)
    if cols[0].button("New", use_container_width=True):
        created = create_session(title="New chat", conn=conn)
        st.session_state["chat_thread_id"] = created.thread_id
        st.session_state.pop("chat_resume_checkpoint_id", None)
        st.session_state.pop("chat_time_travel_hint_checkpoint_id", None)
        with contextlib.suppress(Exception):
            st.query_params.pop("branch", None)
            st.query_params.pop("checkpoint", None)
            st.query_params["chat"] = created.thread_id
        st.rerun()

    confirm_key = f"delete_confirm:{active.thread_id}"
    confirm_delete = st.checkbox("Confirm delete (recoverable)", key=confirm_key)
    if cols[1].button("Delete", use_container_width=True, disabled=not confirm_delete):
        soft_delete_session(conn, thread_id=active.thread_id)
        remaining = list_sessions(conn)
        if remaining:
            st.session_state["chat_thread_id"] = remaining[0].thread_id
        else:
            st.session_state.pop("chat_thread_id", None)
        st.session_state.pop("chat_resume_checkpoint_id", None)
        st.session_state.pop("chat_time_travel_hint_checkpoint_id", None)
        with contextlib.suppress(Exception):
            st.query_params.pop("branch", None)
            st.query_params.pop("checkpoint", None)
        st.session_state.pop(confirm_key, None)
        st.rerun()


def _handle_rename(conn: sqlite3.Connection, active: ChatSession) -> None:
    """Handles session title updates via user input.

    Args:
        conn: SQLite connection.
        active: The currently active session record.
    """
    new_title = st.text_input("Rename", value=active.title, key="chat_session_rename")
    if (
        new_title
        and new_title != active.title
        and st.button("Save name", key="chat_session_rename_save")
    ):
        rename_session(conn, thread_id=active.thread_id, title=new_title)
        st.rerun()


def _handle_purge(conn: sqlite3.Connection, active: ChatSession) -> None:
    """Handles irreversible session purging with confirmation.

    Args:
        conn: SQLite connection.
        active: The currently active session record.
    """
    confirm_key = f"purge_confirm:{active.thread_id}"
    confirm_purge = st.checkbox("I understand this is irreversible", key=confirm_key)
    if st.button(
        "Purge session (hard delete)",
        type="primary",
        disabled=not confirm_purge,
    ):
        purge_session(conn, thread_id=active.thread_id)
        st.session_state.pop("chat_thread_id", None)
        st.session_state.pop("chat_resume_checkpoint_id", None)
        st.session_state.pop("chat_time_travel_hint_checkpoint_id", None)
        with contextlib.suppress(Exception):
            st.query_params.pop("branch", None)
            st.query_params.pop("checkpoint", None)
        st.session_state.pop(confirm_key, None)
        st.rerun()


def render_time_travel_sidebar(
    *,
    coord: ForkableCoordinator,
    conn: sqlite3.Connection,
    thread_id: str,
    user_id: str,
    checkpoints: list[dict[str, object]],
) -> None:
    """Renders time-travel controls for checkpoints and branching.

    Allows users to resume conversation threads from historical state snapshots.

    Args:
        coord: MultiAgentCoordinator instance for forking state.
        conn: SQLite connection for metadata updates.
        thread_id: Target thread identifier.
        user_id: User/tenant namespace.
        checkpoints: List of available checkpoint metadata dictionaries.
    """
    with st.sidebar:
        st.subheader("Time travel")
        st.caption("Resume from a prior checkpoint (creates a fork).")
        ids = [str(c.get("checkpoint_id") or "") for c in checkpoints]
        ids = [x for x in ids if x]
        if not ids:
            st.caption("No checkpoints yet.")
            return

        widget_key = f"chat_time_travel_checkpoint__{thread_id}"
        hint = str(st.session_state.get("chat_time_travel_hint_checkpoint_id") or "")
        if widget_key not in st.session_state and hint in ids:
            st.session_state[widget_key] = hint

        picked = st.selectbox(
            "Checkpoint",
            options=ids,
            index=0,
            key=widget_key,
        )
        if st.button(
            "Resume from checkpoint", key=f"chat_time_travel_resume__{thread_id}"
        ):
            fork = getattr(coord, "fork_from_checkpoint", None)
            if not callable(fork):
                st.error("Time travel is unavailable (missing coordinator support).")
                return

            new_checkpoint_id = fork(
                thread_id=thread_id, user_id=user_id, checkpoint_id=str(picked)
            )
            if not new_checkpoint_id:
                st.error("Failed to fork from the selected checkpoint.")
                return

            touch_session(
                conn, thread_id=thread_id, last_checkpoint_id=str(new_checkpoint_id)
            )
            st.session_state["chat_time_travel_hint_checkpoint_id"] = str(
                new_checkpoint_id
            )
            with contextlib.suppress(Exception):
                st.query_params["chat"] = str(thread_id)
                st.query_params["branch"] = str(new_checkpoint_id)
            st.rerun()
